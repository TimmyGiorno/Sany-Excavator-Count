import sys
import io
import time
import numpy as np
import torch
from PIL import Image
import cv2
from ultralytics import YOLO
import functools

# 强制将标准输出和错误输出设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ========== 强制实时输出 ==========
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print = functools.partial(print, flush=True)


# =================================
class VideoTracker:
    def __init__(self, video_path, model_path, output_path, tracker_config="bytetrack.yaml"):
        """
        初始化视频跟踪器
        """
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.tracker_config = tracker_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ================= 初始化为空闲状态 =================
        self.total_truck_count = 0
        self.total_bucket_count = 0
        self.bucket_full = False
        self.dumping_active = False

        # 统计相关
        self.stat_frames_remaining = 0
        self.ratio_buffer = []
        self.last_avg_ratio = None
        self.is_statting = False
        self.stat_frame_count = 0
        self._last_dumping_box = None
        self._last_dumping_bucket_xyxy = None
        self.dumping_frame_count = 0
        self.retry_count = 0  # 当前重试次数
        self.max_retry_count = 5  # 最大重试次数

        # 稳定期相关
        self.stable_frames_remaining = 0
        self.current_truck_xyxy = None
        self.stat_window_frames = 10
        self.decline_threshold = 0.75

        # ================= 业务逻辑属性 =================
        self.ticket_id = "WAITING"
        self.current_truck_buckets = 0

        # 挂起机制属性
        self.pending_buckets = 0
        self.frames_since_bucket_empty = 0

        self.timeout_sec = 60.0
        self.last_action_time = time.time()

        # 加载模型
        print(f">>> 正在加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)

        self.cap = None
        self.out = None
        self.width = None
        self.height = None
        self.fps = None
        self.total_frames = None


    def get_counts(self):
        return self.total_truck_count, self.current_truck_buckets, self.ticket_id

    @staticmethod
    def _expand_bbox_for_truck(bbox, frame_shape):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width > height:
            new_height = width
            new_y1 = max(0, y2 - new_height)
            new_y2 = y2
            new_x1 = x1
            new_x2 = x2
        else:
            new_width = height
            total_dx = new_width - width
            left_dx = total_dx // 2
            right_dx = total_dx - left_dx
            new_x1 = max(0, x1 - left_dx)
            new_x2 = min(w, x2 + right_dx)
            new_y1 = y1
            new_y2 = y2

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

    def _update_state_machine(self, yolo_results, frame):
        now = time.time()
        self.frames_since_bucket_empty += 1

        # 超时仅模拟数据推送，不清空车辆状态 =================
        if (self.current_truck_buckets > 0 or self.pending_buckets > 0) and (
                now - self.last_action_time > self.timeout_sec):

            print(
                f"  [超时业务事件] 超过 {self.timeout_sec} 秒无装载，向数据库推送进度 - 票号: {self.ticket_id}, "
                f"已装: {self.current_truck_buckets} 铲, 挂起: {self.pending_buckets} 铲")

            self.last_action_time = now

        if yolo_results.boxes is None:
            return

        boxes = yolo_results.boxes
        class_ids = boxes.cls.int().tolist()
        track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(class_ids)
        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()

        truck_boxes = []
        bucket_boxes = []
        dumping_boxes = []
        mine_boxes = []

        for i, (class_id, track_id, xyxy, conf) in enumerate(zip(class_ids, track_ids, xyxy_list, conf_list)):
            if class_id == 2:
                truck_boxes.append(
                    {'class_id': class_id, 'track_id': track_id, 'xyxy': xyxy, 'conf': conf, 'class_name': 'truck'})
            elif class_id == 5:
                mine_boxes.append(
                    {'class_id': class_id, 'track_id': track_id, 'xyxy': xyxy, 'conf': conf, 'class_name': 'mine'})
            elif class_id in [0, 1]:
                bucket_boxes.append({'class_id': class_id, 'track_id': track_id, 'xyxy': xyxy, 'conf': conf,
                                     'class_name': 'bucket-full' if class_id == 1 else 'bucket-empty'})
            elif class_id == 4:
                dumping_boxes.append(
                    {'class_id': class_id, 'track_id': track_id, 'xyxy': xyxy, 'conf': conf, 'class_name': 'dumping'})

        # 1. 处理 bucket 状态转换
        if bucket_boxes:
            best_bucket = max(bucket_boxes, key=lambda x: x['conf'])
            class_id = best_bucket['class_id']

            if class_id == 1:
                # 【加速释放】如果重新变满，说明上一铲绝非切车，立刻把数字显现出来
                if self.pending_buckets > 0:
                    self.current_truck_buckets += self.pending_buckets
                    print(
                        f"  [加速释放] 铲斗重新变满，直接确认上一铲。票号: {self.ticket_id} 铲数: {self.current_truck_buckets}")
                    self.pending_buckets = 0

                if not self.bucket_full:
                    has_overlap = False
                    for truck in truck_boxes:
                        if self._check_horizontal_overlap(best_bucket['xyxy'], truck['xyxy']):
                            has_overlap = True
                            break
                    if not has_overlap:
                        self.bucket_full = True

            elif class_id == 0:
                if self.bucket_full:
                    has_overlap = False
                    for truck in truck_boxes:
                        if self._check_horizontal_overlap(best_bucket['xyxy'], truck['xyxy']):
                            has_overlap = True
                            break

                    if has_overlap:
                        self.bucket_full = False
                        self.total_bucket_count += 1

                        self.pending_buckets += 1
                        self.frames_since_bucket_empty = 0
                        self.last_action_time = time.time()

                        if self.ticket_id == "WAITING":
                            self.total_truck_count += 1
                            self.ticket_id = f"TKT_{int(time.time() * 1000)}"
                            print(f"  [业务事件] 新车入场，生成票号 {self.ticket_id}")

                        print(f"  [业务事件] 单铲装载完成！进入快速确认期 (挂起 {self.pending_buckets} 铲)")
                    else:
                        self.bucket_full = False

        # 2. 处理 dumping 状态跟踪
        has_dumping = len(dumping_boxes) > 0

        if has_dumping:
            self._last_dumping_box = dumping_boxes[0]['xyxy']
            self.dumping_frame_count += 1
            if not self.dumping_active:
                if self.dumping_frame_count >= 5:
                    self.dumping_active = True

            if self.stable_frames_remaining > 0 or self.is_statting:
                self.stable_frames_remaining = 0
                self.is_statting = False
                self.stat_frames_remaining = 0
                self.ratio_buffer = []
                self.current_truck_xyxy = None
                self._last_dumping_bucket_xyxy = None

        elif not has_dumping:
            if self.dumping_frame_count > 0:
                self.dumping_frame_count = 0

            if self.dumping_active:
                self.dumping_active = False
                self.stable_frames_remaining = 1

                if bucket_boxes:
                    best_bucket = max(bucket_boxes, key=lambda x: x['conf'])
                    self._last_dumping_bucket_xyxy = best_bucket['xyxy']
                else:
                    self._last_dumping_bucket_xyxy = None

        # 3. 处理稳定期
        if self.stable_frames_remaining > 0:
            self.stable_frames_remaining -= 1

            if self.stable_frames_remaining == 0:

                # 如果没有 bucket 位置记录，尝试从当前帧获取
                if self._last_dumping_bucket_xyxy is None and bucket_boxes:
                    best_bucket = max(bucket_boxes, key=lambda x: x['conf'])
                    self._last_dumping_bucket_xyxy = best_bucket['xyxy']

                if self._last_dumping_bucket_xyxy is not None:
                    if truck_boxes:
                        # 找到 truck，开始统计
                        self.is_statting = True
                        self.ratio_buffer = []
                        self.stat_frames_remaining = 10
                        self.retry_count = 0  # 重置重试计数

                        # 找与 bucket 水平距离最近的 truck
                        min_distance = float('inf')
                        bucket_center_x = (self._last_dumping_bucket_xyxy[0] + self._last_dumping_bucket_xyxy[2]) / 2

                        for truck in truck_boxes:
                            truck_center_x = (truck['xyxy'][0] + truck['xyxy'][2]) / 2
                            distance = abs(truck_center_x - bucket_center_x)
                            if distance < min_distance:
                                min_distance = distance
                                self.current_truck_xyxy = truck['xyxy']

                        # 清理 bucket 位置记录
                        self._last_dumping_bucket_xyxy = None
                    else:
                        # 没有 truck，重试
                        self.retry_count += 1
                        if self.retry_count < self.max_retry_count:
                            self.stable_frames_remaining = 1
                        else:
                            # 达到最大重试次数，放弃本次统计
                            self.retry_count = 0
                            self._last_dumping_bucket_xyxy = None
                else:
                    # 没有 bucket 位置记录，且当前帧也没有 bucket
                    self.retry_count += 1
                    if self.retry_count < self.max_retry_count:
                        self.stable_frames_remaining = 1
                    else:
                        # 达到最大重试次数，放弃本次统计
                        self.retry_count = 0
                        self._last_dumping_bucket_xyxy = None

        # 4. 统计 mine/truck 面积比值并结算挂起的铲数
        if self.is_statting and self.stat_frames_remaining > 0:
            self.stat_frames_remaining -= 1
            self.stat_frame_count += 1

            if self.current_truck_xyxy is not None and truck_boxes:
                min_distance = float('inf')
                current_truck = None
                prev_center_x = (self.current_truck_xyxy[0] + self.current_truck_xyxy[2]) / 2

                for truck in truck_boxes:
                    truck_center_x = (truck['xyxy'][0] + truck['xyxy'][2]) / 2
                    distance = abs(truck_center_x - prev_center_x)
                    if distance < min_distance:
                        min_distance = distance
                        current_truck = truck

                if current_truck is not None:
                    self.current_truck_xyxy = current_truck['xyxy']
                    truck_x1, truck_y1, truck_x2, truck_y2 = self.current_truck_xyxy
                    truck_area = (truck_x2 - truck_x1) * (truck_y2 - truck_y1)

                    if truck_area > 0:
                        mine_area = 0
                        for mine in mine_boxes:
                            if self._check_horizontal_overlap(self.current_truck_xyxy, mine['xyxy']):
                                mine_x1, mine_y1, mine_x2, mine_y2 = mine['xyxy']
                                current_mine_area = (mine_x2 - mine_x1) * (mine_y2 - mine_y1)
                                if current_mine_area > mine_area:
                                    mine_area = current_mine_area
                        ratio = mine_area / truck_area
                    else:
                        ratio = 0.0
                else:
                    ratio = 0.0
            else:
                ratio = 0.0

            self.ratio_buffer.append(ratio)

            if self.stat_frames_remaining == 0:
                self.is_statting = False
                avg_ratio = sum(self.ratio_buffer) / len(self.ratio_buffer) if self.ratio_buffer else 0.0

                if self.last_avg_ratio is None:
                    self.last_avg_ratio = avg_ratio
                    if self.pending_buckets > 0:
                        self.current_truck_buckets += self.pending_buckets
                        self.pending_buckets = 0
                else:
                    if self.last_avg_ratio == 0.0 and avg_ratio == 0.0:
                        if self.pending_buckets > 0:
                            self.current_truck_buckets += self.pending_buckets
                            self.pending_buckets = 0
                    else:
                        decline = (
                            self.last_avg_ratio - avg_ratio) / self.last_avg_ratio if self.last_avg_ratio > 0 else 0.0

                        if avg_ratio == 0.0 or decline >= self.decline_threshold:
                            self.total_truck_count += 1
                            self.ticket_id = f"TKT_{int(time.time() * 1000)}"
                            self.current_truck_buckets = self.pending_buckets
                            self.pending_buckets = 0
                            self.last_action_time = time.time()
                        else:
                            if self.pending_buckets > 0:
                                self.current_truck_buckets += self.pending_buckets
                                self.pending_buckets = 0

                    self.last_avg_ratio = avg_ratio

                self.current_truck_xyxy = None
                self._last_dumping_bucket_xyxy = None

        # 5. 快速强行确认 (降至 15 帧，约0.5-0.6秒)
        if self.pending_buckets > 0 and self.frames_since_bucket_empty > 15:
            self.current_truck_buckets += self.pending_buckets
            print(f"  [快速强行确认] 自动合并。票号 {self.ticket_id} 铲数: {self.current_truck_buckets}")
            self.pending_buckets = 0

    @staticmethod
    def _check_horizontal_overlap(box1, box2):
        x1_min, x1_max = box1[0], box1[2]
        x2_min, x2_max = box2[0], box2[2]
        return not (x1_max < x2_min or x2_max < x1_min)

    def run_video_inference(self):
        print(f">>> 正在打开视频: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频，请检查路径！")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0

        print(">>> 开始逐帧推断并渲染...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            results = self.model.track(frame, persist=True, tracker=self.tracker_config, verbose=False)

            self._update_state_machine(results[0], frame)

            trucks, buckets, tkt_id = self.get_counts()

            annotated_frame = results[0].plot()

            ui_x1, ui_y1 = self.width - 400, 20
            ui_x2, ui_y2 = self.width - 20, 160
            cv2.rectangle(annotated_frame, (ui_x1, ui_y1), (ui_x2, ui_y2), (0, 0, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_frame, f"Trucks:  {trucks}", (self.width - 380, 60), font, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Buckets: {buckets}", (self.width - 380, 105), font, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Ticket:  {tkt_id}", (self.width - 380, 145), font, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA)

            self.out.write(annotated_frame)

            if frame_count % 50 == 0:
                print(f"进度: {frame_count} / {self.total_frames} 帧...")

        self.cap.release()
        self.out.release()
        print(f"\n>>> 推断完成！输出视频已保存至: {self.output_path}")


if __name__ == "__main__":
    TEST_VIDEO = "./tmp_files/bug_1.mp4"
    TRAINED_MODEL = "./tmp_files/best.pt"
    OUTPUT_VIDEO = "./tmp_files/test5.mp4"

    tracker = VideoTracker(
        video_path=TEST_VIDEO,
        model_path=TRAINED_MODEL,
        output_path=OUTPUT_VIDEO,
        tracker_config="bytetrack.yaml"
    )

    tracker.run_video_inference()
