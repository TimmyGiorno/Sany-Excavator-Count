import sys
import io
import time
import json
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import functools

# 强制将标准输出和错误输出设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ========== 强制实时输出 ==========
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print = functools.partial(print, flush=True)


class VideoTracker:
    def __init__(self, video_path, model_path, output_path, tracker_config="bytetrack.yaml"):
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.tracker_config = tracker_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ================= 业务参数配置 =================
        self.timeout_ms = 60000  # 业务超时时间：1 分钟没动作就算超时
        self.decline_threshold = 0.75
        self.min_mineral_ratio = 0.15  # 矿物面积比例阈值，小于该值强制归零

        # ================= 服务器事件结构列表 =================
        self.pending_bucket_events = []
        self.pending_truck_events = []

        # ================= 状态机与缓存区 =================
        self.ticket_id = "WAITING"
        self.total_truck_count = 0
        self.total_bucket_count = 0
        self.current_truck_buckets = 0

        self.pending_buckets = 0
        self.frames_since_bucket_empty = 0
        self.pending_queue = []  # 记录挂起铲时间戳的队列

        self.has_pushed_timeout = False
        self.timeout_bucket_count = -1

        self.bucket_full = False

        self.full_confirm_frames = 0
        self.empty_confirm_frames = 0

        self.dumping_active = False
        self.dumping_frame_count = 0
        self.dumping_lost_frames = 0

        self.retry_count = 0
        self.max_retry_count = 5

        self.current_dump_start_time = 0
        self.truck_load_start_time = 0
        self.truck_load_end_time = 0
        self.last_dump_end_time = 0
        self.last_action_time = 0

        self.is_truck_active = False

        self.pending_bucket_secured = False
        self.secured_dump_start_time = 0

        self.current_truck_xyxy = None
        self._last_dumping_bucket_xyxy = None

        self.stable_frames_remaining = 0
        self.is_statting = False
        self.stat_frames_remaining = 0
        self.ratio_buffer = []
        self.last_avg_ratio = -1.0

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
        display_buckets = self.current_truck_buckets + self.pending_buckets
        return self.total_truck_count, display_buckets, self.ticket_id, self.last_avg_ratio, self.bucket_full

    @staticmethod
    def _check_horizontal_overlap(box1, box2):
        x1_min, x1_max = box1[0], box1[2]
        x2_min, x2_max = box2[0], box2[2]
        return not (x1_max < x2_min or x2_max < x1_min)

    # ================= C++ 工具函数镜像 =================
    def _force_complete_truck(self, completed_type=0):
        if self.is_truck_active:
            should_push_event = True

            if completed_type == 1:
                self.timeout_bucket_count = self.current_truck_buckets
            elif completed_type == 0:
                if self.timeout_bucket_count != -1 and self.timeout_bucket_count == self.current_truck_buckets:
                    should_push_event = False

            if should_push_event:
                end_time = self.truck_load_end_time if self.truck_load_end_time > 0 else self.last_action_time
                if end_time <= 0:
                    end_time = int(time.time() * 1000)

                self.pending_truck_events.append({
                    "ticket_id": self.ticket_id,
                    "total_truck_count": self.total_truck_count,
                    "total_bucket_count": self.current_truck_buckets,
                    "load_start_time": self.truck_load_start_time,
                    "load_end_time": end_time,
                    "completed_type": completed_type
                })

            if completed_type == 0:
                self.is_truck_active = False
                self.ticket_id = "WAITING"
                self.current_truck_buckets = 0
                self.has_pushed_timeout = False
                self.timeout_bucket_count = -1
                self.last_avg_ratio = -1.0

    def _commit_pending_buckets(self):
        for pb in self.pending_queue:
            self.current_truck_buckets += 1
            self.truck_load_end_time = pb['dump_end_time']

            mineral_ratio = 0.0 if self.last_avg_ratio < 0 else self.last_avg_ratio

            self.pending_bucket_events.append({
                "ticket_id": self.ticket_id,
                "total_truck_count": self.total_truck_count,
                "current_bucket_count": self.current_truck_buckets,
                "dump_start_time": pb['dump_start_time'],
                "dump_end_time": pb['dump_end_time'],
                "last_mineral_ratio": mineral_ratio
            })

        self.pending_buckets = 0
        self.pending_queue.clear()

    def _cut_truck(self, now_ms, new_ratio=0.0):
        self._force_complete_truck(0)
        self.is_truck_active = True
        self.total_truck_count += 1
        self.ticket_id = f"TKT_{int(time.time() * 1000)}"
        self.current_truck_buckets = 0
        self.has_pushed_timeout = False
        self.last_action_time = now_ms

        if self.pending_queue:
            self.truck_load_start_time = self.pending_queue[0]['dump_start_time']
        else:
            self.truck_load_start_time = now_ms

        self.last_avg_ratio = new_ratio
        self._commit_pending_buckets()

    def _consume_and_print_events(self):
        if not self.pending_truck_events and not self.pending_bucket_events:
            return

        for ev in self.pending_truck_events:
            c_type = "超时强制结束" if ev['completed_type'] == 1 else "正常开走完结"
            msg = (f'{{"类型": "车辆完结事件", "完结方式": "{c_type}", '
                   f'"票号": "{ev["ticket_id"]}", "总装车数": {ev["total_truck_count"]}, '
                   f'"总共铲斗数": {ev["total_bucket_count"]}, "装车开始时间": {ev["load_start_time"]}, '
                   f'"装车结束时间": {ev["load_end_time"]}}}')
            print(msg)

        for ev in self.pending_bucket_events:
            msg = (f'{{"类型": "铲斗事件", "票号": "{ev["ticket_id"]}", '
                   f'"总装车数": {ev["total_truck_count"]}, "当前铲斗数": {ev["current_bucket_count"]}, '
                   f'"矿物占比": {ev["last_mineral_ratio"]:.6f}, "卸料开始时间": {ev["dump_start_time"]}, '
                   f'"卸料结束时间": {ev["dump_end_time"]}}}')
            print(msg)

        self.pending_truck_events.clear()
        self.pending_bucket_events.clear()

    def _trigger_bucket_count(self, now_ms):
        self.pending_bucket_secured = False
        self.total_bucket_count += 1
        self.pending_buckets += 1
        self.frames_since_bucket_empty = 0
        self.last_action_time = now_ms
        self.has_pushed_timeout = False

        if not self.is_truck_active:
            self.is_truck_active = True
            self.total_truck_count += 1
            self.ticket_id = f"TKT_{int(time.time() * 1000)}"
            self.current_truck_buckets = 0
            self.truck_load_start_time = self.secured_dump_start_time if self.secured_dump_start_time > 0 else now_ms

        pb_start = self.secured_dump_start_time if self.secured_dump_start_time > 0 else now_ms
        self.pending_queue.append({
            "dump_start_time": pb_start,
            "dump_end_time": now_ms
        })
        self.current_dump_start_time = 0
        self.stable_frames_remaining = 1

    # ================= 核心状态机逻辑 =================
    def _update_state_machine(self, yolo_results, frame):
        now_ms = int(time.time() * 1000)
        self.frames_since_bucket_empty += 1

        frame_h, frame_w = frame.shape[:2]

        if self.last_action_time > 0 and (self.current_truck_buckets > 0 or self.pending_buckets > 0) and (
                now_ms - self.last_action_time > self.timeout_ms):
            if not self.has_pushed_timeout:
                self.has_pushed_timeout = True
                self._force_complete_truck(1)

        if yolo_results.boxes is None:
            return

        boxes = yolo_results.boxes
        class_ids = boxes.cls.int().tolist()
        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()

        truck_boxes, bucket_boxes, dumping_boxes, mine_boxes, loading_boxes = [], [], [], [], []

        for class_id, xyxy, conf in zip(class_ids, xyxy_list, conf_list):
            x1, y1, x2, y2 = xyxy

            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(0, min(frame_w - 1, x2))
            y2 = max(0, min(frame_h - 1, y2))

            bw = x2 - x1
            bh = y2 - y1

            box_dict = {'class_id': class_id, 'xyxy': [x1, y1, x2, y2], 'conf': conf, 'w': bw, 'h': bh}

            if class_id == 2:
                truck_boxes.append(box_dict)
            elif class_id in [0, 1]:
                if bw > 100 and bh > 100:
                    bucket_boxes.append(box_dict)
            elif class_id == 3:
                loading_boxes.append(box_dict)
            elif class_id == 4:
                dumping_boxes.append(box_dict)
            elif class_id == 5:
                mine_boxes.append(box_dict)

        main_truck = max(truck_boxes, key=lambda t: t['w'] * t['h']) if truck_boxes else None
        best_bucket = max(bucket_boxes, key=lambda b: b['w'] * b['h']) if bucket_boxes else None

        # ============================== 1. 铲斗变满判定 (直球模式) ==============================
        is_digging = False
        if loading_boxes or (best_bucket and best_bucket['class_id'] == 1):
            is_digging = True

        if is_digging and not self.dumping_active and not dumping_boxes:
            self.empty_confirm_frames = 0
            if not self.bucket_full:
                self.full_confirm_frames += 1
                if self.full_confirm_frames >= 3:
                    self.bucket_full = True
                    self.full_confirm_frames = 0
        else:
            self.full_confirm_frames = 0

        # ============================== 2. 卸矿动作追踪 (绝对权威 + 空间立体防御) ==============================
        has_dumping = len(dumping_boxes) > 0

        if has_dumping:
            self.dumping_frame_count += 1
            self.dumping_lost_frames = 0

            if self.dumping_frame_count == 1:
                self.current_dump_start_time = now_ms

            if not self.dumping_active and self.dumping_frame_count >= 2:
                self.dumping_active = True

                if best_bucket:
                    self._last_dumping_bucket_xyxy = best_bucket['xyxy']
                else:
                    self._last_dumping_bucket_xyxy = dumping_boxes[0]['xyxy']

                # 【新增空间立体防御】：甄别是“往车里倒”还是“往车旁边的地上倒”
                is_dumping_inside_truck = True  # 兜底信任

                if main_truck:
                    # 1. 基础 X 轴重叠 (排除在卡车左右很远的地方倒土)
                    overlap_x = self._check_horizontal_overlap(self._last_dumping_bucket_xyxy, main_truck['xyxy'])

                    if overlap_x:
                        # 2. 如果水平重合，必须检查高度！
                        # 真正的往车里倒矿，铲斗的顶部必须高于卡车高度的一半 (防倒在地上重叠的视觉错觉)
                        d_x1, d_y1, d_x2, d_y2 = self._last_dumping_bucket_xyxy
                        t_x1, t_y1, t_x2, t_y2 = main_truck['xyxy']
                        t_h = t_y2 - t_y1

                        is_high_enough = d_y1 < (t_y1 + t_h * 0.5)
                        is_dumping_inside_truck = is_high_enough
                    else:
                        # 完全没水平重叠，肯定是在外边倒
                        is_dumping_inside_truck = False

                if self.bucket_full:
                    if is_dumping_inside_truck:
                        self.bucket_full = False
                        self.pending_bucket_secured = True
                        self.secured_dump_start_time = self.current_dump_start_time if self.current_dump_start_time > 0 else now_ms
                    else:
                        pass  # 触发立体拦截，不记账

            if self.stable_frames_remaining > 0 or self.is_statting:
                self.stable_frames_remaining = 0
                self.is_statting = False
                self.stat_frames_remaining = 0
                self.ratio_buffer.clear()
                self.current_truck_xyxy = None

        else:
            if self.dumping_active:
                self.dumping_lost_frames += 1
                if self.dumping_lost_frames > 6:
                    self.dumping_active = False
                    self.dumping_frame_count = 0
                    self.dumping_lost_frames = 0

                    if self.pending_bucket_secured:
                        self._trigger_bucket_count(now_ms)

            # 兜底机制：出了空斗黄框并在车上
            elif best_bucket and best_bucket['class_id'] == 0 and self.bucket_full and main_truck:
                if self._check_horizontal_overlap(best_bucket['xyxy'], main_truck['xyxy']):
                    self.empty_confirm_frames += 1

                    if self.empty_confirm_frames >= 5:
                        self.bucket_full = False
                        self.empty_confirm_frames = 0
                        self._last_dumping_bucket_xyxy = best_bucket['xyxy']
                        self.secured_dump_start_time = now_ms - 500
                        self.pending_bucket_secured = True
                        self._trigger_bucket_count(now_ms)
                else:
                    self.empty_confirm_frames = 0

        # ============================== 3. 寻找用于计算比值的卡车 ==============================
        if self.stable_frames_remaining > 0:
            self.stable_frames_remaining -= 1
            if self.stable_frames_remaining == 0:
                if truck_boxes:
                    self.is_statting = True
                    self.ratio_buffer.clear()
                    self.stat_frames_remaining = 15
                    self.retry_count = 0

                    if self._last_dumping_bucket_xyxy is not None:
                        min_distance = float('inf')
                        b_cx = (self._last_dumping_bucket_xyxy[0] + self._last_dumping_bucket_xyxy[2]) / 2.0
                        for truck in truck_boxes:
                            t_cx = (truck['xyxy'][0] + truck['xyxy'][2]) / 2.0
                            dist = abs(t_cx - b_cx)
                            if dist < min_distance:
                                min_distance = dist
                                self.current_truck_xyxy = truck['xyxy']
                    else:
                        best_t = max(truck_boxes, key=lambda t: t['w'] * t['h'])
                        self.current_truck_xyxy = best_t['xyxy']

                    self._last_dumping_bucket_xyxy = None
                else:
                    self.retry_count += 1
                    if self.retry_count < self.max_retry_count:
                        self.stable_frames_remaining = 1
                    else:
                        self.retry_count = 0
                        self._last_dumping_bucket_xyxy = None
                        if self.last_avg_ratio < 0:
                            self.last_avg_ratio = 0.0
                        if self.pending_buckets > 0:
                            self._commit_pending_buckets()

        # ============================== 4. 真实比例断崖下跌计算 ==============================
        if self.is_statting and self.stat_frames_remaining > 0:
            self.stat_frames_remaining -= 1

            if self.current_truck_xyxy is not None and truck_boxes:
                min_dist = float('inf')
                best_t_xyxy = None
                prev_cx = (self.current_truck_xyxy[0] + self.current_truck_xyxy[2]) / 2.0

                for truck in truck_boxes:
                    t_cx = (truck['xyxy'][0] + truck['xyxy'][2]) / 2.0
                    dist = abs(t_cx - prev_cx)
                    if dist < min_dist:
                        min_dist = dist
                        best_t_xyxy = truck['xyxy']

                if best_t_xyxy is not None:
                    self.current_truck_xyxy = best_t_xyxy
                    tx1, ty1, tx2, ty2 = best_t_xyxy
                    truck_area = (tx2 - tx1) * (ty2 - ty1)
                    max_mine_area = 0.0

                    for mine in mine_boxes:
                        if self._check_horizontal_overlap(best_t_xyxy, mine['xyxy']):
                            mx1, my1, mx2, my2 = mine['xyxy']
                            m_area = (mx2 - mx1) * (my2 - my1)
                            if m_area > max_mine_area:
                                max_mine_area = m_area
                    self.ratio_buffer.append(max_mine_area / truck_area)
                else:
                    self.ratio_buffer.append(0.0)
            else:
                self.ratio_buffer.append(0.0)

            if self.stat_frames_remaining == 0:
                self.is_statting = False

                valid_ratios = [r for r in self.ratio_buffer if r > 0.0]
                avg_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0

                if avg_ratio < self.min_mineral_ratio:
                    avg_ratio = 0.0

                if self.last_avg_ratio < 0:
                    self.last_avg_ratio = avg_ratio
                    if self.pending_buckets > 0:
                        self._commit_pending_buckets()
                else:
                    if self.last_avg_ratio == 0.0 and avg_ratio == 0.0:
                        self.last_avg_ratio = avg_ratio
                        if self.pending_buckets > 0:
                            self._commit_pending_buckets()
                    else:
                        decline = (
                                              self.last_avg_ratio - avg_ratio) / self.last_avg_ratio if self.last_avg_ratio > 0 else 0.0

                        if avg_ratio == 0.0 or decline >= self.decline_threshold:
                            self._cut_truck(now_ms, avg_ratio)
                        else:
                            self.last_avg_ratio = avg_ratio
                            if self.pending_buckets > 0:
                                self._commit_pending_buckets()

                self.current_truck_xyxy = None

        # ============================== 5. 快速强行合并兜底 ==============================
        if self.pending_buckets > 0 and self.frames_since_bucket_empty > 60 and not self.is_statting and not self.dumping_active:
            if self.last_avg_ratio < 0:
                self.last_avg_ratio = 0.0
            self._commit_pending_buckets()

    def run_video_inference(self, target_fps=10.0):
        print(f">>> 正在打开视频: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频，请检查路径！")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        orig_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_step = max(1, int(round(orig_fps / target_fps)))
        actual_out_fps = orig_fps / frame_step

        print(
            f">>> 视频原始 FPS: {orig_fps:.1f} | 模拟目标 FPS: {target_fps:.1f} | 抽帧步长: 每 {frame_step} 帧推理 1 帧")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, actual_out_fps, (self.width, self.height))

        raw_frame_count = 0
        infer_frame_count = 0
        total_time_ms = 0

        print(">>> 开始逐帧推断并渲染...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            raw_frame_count += 1

            if raw_frame_count % frame_step != 0:
                continue

            infer_frame_count += 1

            start_time = time.time()

            results = self.model.track(frame, persist=True, tracker=self.tracker_config, verbose=False)

            # 【核心逻辑触发】
            self._update_state_machine(results[0], frame)

            # 【输出 C++ 镜像事件到控制台】
            self._consume_and_print_events()

            end_time = time.time()
            total_time_ms += (end_time - start_time) * 1000

            trucks, buckets, tkt_id, ratio, is_full = self.get_counts()

            # UI 渲染
            annotated_frame = results[0].plot()

            ui_x1, ui_y1 = self.width - 400, 20
            ui_x2, ui_y2 = self.width - 20, 240
            cv2.rectangle(annotated_frame, (ui_x1, ui_y1), (ui_x2, ui_y2), (0, 0, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            ratio_str = "0.00" if ratio < 0 else f"{ratio:.2f}"

            cv2.putText(annotated_frame, f"Trucks:  {trucks}", (self.width - 380, 60), font, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Buckets: {buckets}", (self.width - 380, 105), font, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Ratio:   {ratio_str}", (self.width - 380, 150), font, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Ticket:  {tkt_id}", (self.width - 380, 190), font, 0.7, (0, 255, 255), 2,
                        cv2.LINE_AA)

            state_str = "FULL" if is_full else "EMPTY"
            state_color = (0, 0, 255) if is_full else (0, 255, 255)
            cv2.putText(annotated_frame, f"State:   {state_str}", (self.width - 380, 230), font, 1.0, state_color, 2,
                        cv2.LINE_AA)

            self.out.write(annotated_frame)

            if infer_frame_count % 100 == 0:
                avg_time_ms = total_time_ms / 100.0
                print(
                    f"📊 [模拟进度] 物理帧: {raw_frame_count} / {self.total_frames} | AI 已处理: {infer_frame_count} 帧 | 平均耗时: {avg_time_ms:.2f} ms")
                total_time_ms = 0

        self.cap.release()
        self.out.release()
        print(f"\n>>> 推断完成！共读取物理帧: {raw_frame_count}，实际推理帧: {infer_frame_count}")
        print(f">>> 输出视频已保存至: {self.output_path}")


if __name__ == "__main__":
    TEST_VIDEO = "./tmp_files/test2.mp4"
    TRAINED_MODEL = "./tmp_files/best.pt"
    OUTPUT_VIDEO = "./tmp_files/test2_output.mp4"

    tracker = VideoTracker(
        video_path=TEST_VIDEO,
        model_path=TRAINED_MODEL,
        output_path=OUTPUT_VIDEO,
        tracker_config="bytetrack.yaml"
    )

    tracker.run_video_inference(target_fps=10.0)