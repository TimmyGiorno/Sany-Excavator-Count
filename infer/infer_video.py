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
        self.timeout_ms = 60000
        self.decline_threshold = 0.75
        self.min_mineral_ratio = 0.15

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
        self.pending_queue = []

        self.has_pushed_timeout = False
        self.timeout_bucket_count = -1

        self.bucket_full = False

        self.full_confirm_frames = 0
        self.empty_confirm_frames = 0

        self.dumping_active = False
        self.dumping_frame_count = 0
        self.dumping_lost_frames = 0

        self.current_dump_start_time = 0
        self.truck_load_start_time = 0
        self.truck_load_end_time = 0
        self.last_dump_end_time = 0
        self.last_action_time = 0

        self.is_truck_active = False

        self.pending_bucket_secured = False
        self.secured_dump_start_time = 0

        self._last_dumping_bucket_xyxy = None

        self.ratio_buffer = []
        self.last_avg_ratio = -1.0

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

        valid_ratios = [r for r in self.ratio_buffer if r > 0.0]
        avg_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0
        self.ratio_buffer.clear()

        if avg_ratio < self.min_mineral_ratio:
            avg_ratio = 0.0

        if self.last_avg_ratio < 0:
            self.last_avg_ratio = avg_ratio
            self._commit_pending_buckets()
        else:
            if self.last_avg_ratio == 0.0 and avg_ratio == 0.0:
                self.last_avg_ratio = avg_ratio
                self._commit_pending_buckets()
            else:
                decline = (self.last_avg_ratio - avg_ratio) / self.last_avg_ratio if self.last_avg_ratio > 0 else 0.0
                if avg_ratio == 0.0 or decline >= self.decline_threshold:
                    self._cut_truck(now_ms, avg_ratio)
                else:
                    self.last_avg_ratio = avg_ratio
                    self._commit_pending_buckets()

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
                bucket_boxes.append(box_dict)
            elif class_id == 4:
                dumping_boxes.append(box_dict)
            elif class_id == 5:
                mine_boxes.append(box_dict)

        if len(bucket_boxes) > 1:
            best_conf_bucket = max(bucket_boxes, key=lambda b: b['conf'])
            bucket_boxes = [best_conf_bucket]

        best_bucket = bucket_boxes[0] if bucket_boxes else None

        if best_bucket:
            b_area = best_bucket['w'] * best_bucket['h']
            truck_boxes = [t for t in truck_boxes if (t['w'] * t['h']) > b_area]

        main_truck = max(truck_boxes, key=lambda t: t['w'] * t['h']) if truck_boxes else None

        # ============================== 1. 铲斗变满判定 ==============================
        is_full_detected = (best_bucket and best_bucket['class_id'] == 1)

        if is_full_detected and not self.dumping_active and not dumping_boxes:
            self.empty_confirm_frames = 0
            if not self.bucket_full:
                self.full_confirm_frames += 1
                if self.full_confirm_frames >= 20:
                    self.bucket_full = True
                    self.full_confirm_frames = 0
                    self.ratio_buffer.clear()  # 防止残留比例干扰
        else:
            self.full_confirm_frames = 0

        # ============================== 2. 卸矿动作追踪 ==============================
        has_dumping = len(dumping_boxes) > 0

        if has_dumping:
            self.dumping_frame_count += 1
            self.dumping_lost_frames = 0

            if self.dumping_frame_count == 1:
                self.current_dump_start_time = now_ms

            if not self.dumping_active and self.dumping_frame_count >= 10:
                self.dumping_active = True

                if best_bucket:
                    self._last_dumping_bucket_xyxy = best_bucket['xyxy']
                else:
                    self._last_dumping_bucket_xyxy = dumping_boxes[0]['xyxy']

                is_dumping_inside_truck = False

                if main_truck:
                    overlap_x = self._check_horizontal_overlap(self._last_dumping_bucket_xyxy, main_truck['xyxy'])
                    if overlap_x:
                        d_x1, d_y1, d_x2, d_y2 = self._last_dumping_bucket_xyxy
                        t_x1, t_y1, t_x2, t_y2 = main_truck['xyxy']
                        is_dumping_inside_truck = (d_y1 < t_y1)
                    else:
                        is_dumping_inside_truck = False

                if self.bucket_full:
                    if is_dumping_inside_truck:
                        self.pending_bucket_secured = True
                        self.secured_dump_start_time = self.current_dump_start_time if self.current_dump_start_time > 0 else now_ms
        else:
            if self.dumping_active:
                self.dumping_lost_frames += 1
                if self.dumping_lost_frames > 15:
                    self.dumping_active = False
                    self.dumping_frame_count = 0
                    self.dumping_lost_frames = 0

                    if self.pending_bucket_secured:
                        self.bucket_full = False
                        self.empty_confirm_frames = 0
                        self._trigger_bucket_count(now_ms)
                        self.pending_bucket_secured = False

        # ============================== 2.5 稳定空斗结算 ==============================
        is_empty_detected = (best_bucket and best_bucket['class_id'] == 0)

        if is_empty_detected and self.bucket_full:
            is_valid_empty = False

            if self.pending_bucket_secured:
                is_valid_empty = True
            elif main_truck:
                overlap_x = self._check_horizontal_overlap(best_bucket['xyxy'], main_truck['xyxy'])
                if overlap_x:
                    b_x1, b_y1, b_x2, b_y2 = best_bucket['xyxy']
                    t_x1, t_y1, t_x2, t_y2 = main_truck['xyxy']
                    is_valid_empty = (b_y1 < t_y1)

            if is_valid_empty:
                self.empty_confirm_frames += 1

                if self.empty_confirm_frames >= 10:
                    self.bucket_full = False
                    self.empty_confirm_frames = 0
                    self._last_dumping_bucket_xyxy = best_bucket['xyxy']

                    if not self.pending_bucket_secured:
                        self.secured_dump_start_time = now_ms - 2000

                    self._trigger_bucket_count(now_ms)
                    self.pending_bucket_secured = False

                    self.dumping_active = False
                    self.dumping_frame_count = 0
                    self.dumping_lost_frames = 0
            else:
                self.empty_confirm_frames = 0
        else:
            if not is_empty_detected:
                self.empty_confirm_frames = 0

        # ============================== 3. 实时跟踪采集核心 ==============================
        # 当正在卸载，或者空斗正在消抖确认期间，就认定是“核心动作捕捉区”
        critical_zone = self.dumping_active or self.pending_bucket_secured or (self.empty_confirm_frames > 0)

        if critical_zone and main_truck:
            frame_area = frame_w * frame_h
            max_mine_area = 0.0

            # 遍历所有与卡车水平重合的矿物，找到最大的
            for mine in mine_boxes:
                if self._check_horizontal_overlap(main_truck['xyxy'], mine['xyxy']):
                    mx1, my1, mx2, my2 = mine['xyxy']
                    m_area = (mx2 - mx1) * (my2 - my1)
                    if m_area > max_mine_area:
                        max_mine_area = m_area

            # 使用全图面积作为分母，彻底免疫卡车被截断的干扰，该数值要乘 3
            self.ratio_buffer.append(3 * max_mine_area / frame_area)

            # 防止长时间停滞导致内存溢出
            if len(self.ratio_buffer) > 100:
                self.ratio_buffer.pop(0)

        # 旧版的老步骤 4 和 5 已经被彻底消灭了，都在 trigger 里完成了瞬间结算。

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

            self._update_state_machine(results[0], frame)

            self._consume_and_print_events()

            end_time = time.time()
            total_time_ms += (end_time - start_time) * 1000

            trucks, buckets, tkt_id, ratio, is_full = self.get_counts()

            # UI 渲染
            annotated_frame = results[0].plot()

            ui_x1, ui_y1 = 20, 20
            ui_x2, ui_y2 = 400, 240
            cv2.rectangle(annotated_frame, (ui_x1, ui_y1), (ui_x2, ui_y2), (0, 0, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            ratio_str = "0.0000" if ratio < 0 else f"{ratio:.4f}"

            cv2.putText(annotated_frame, f"Trucks:  {trucks}", (40, 60), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Buckets: {buckets}", (40, 105), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Ratio:   {ratio_str}", (40, 150), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Ticket:  {tkt_id}", (40, 190), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

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
    TEST_VIDEO = "./tmp_files/test5.mp4"
    TRAINED_MODEL = "./tmp_files/best.pt"
    OUTPUT_VIDEO = "./tmp_files/test5_output.mp4"

    tracker = VideoTracker(
        video_path=TEST_VIDEO,
        model_path=TRAINED_MODEL,
        output_path=OUTPUT_VIDEO,
        tracker_config="bytetrack.yaml"
    )

    tracker.run_video_inference(target_fps=10.0)