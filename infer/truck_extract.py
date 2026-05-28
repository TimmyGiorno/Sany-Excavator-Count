import sys
import io
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import json
import traceback
import time
import os

# ========== 强制实时输出 ==========
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import functools

print = functools.partial(print, flush=True)


class DumpingTruckExtractor:
    def __init__(self, model_path, tracker_config="bytetrack.yaml",
                 frame_buffer=5, min_dumping_frames=10):
        """
        初始化提取器

        Args:
            model_path: 模型权重路径
            tracker_config: 跟踪器配置文件路径
            frame_buffer: 状态变化时的缓冲帧数（防止抖动）
            min_dumping_frames: 最小dumping持续帧数（过滤误检）
        """
        self.model_path = model_path
        self.tracker_config = tracker_config
        self.frame_buffer = frame_buffer
        self.min_dumping_frames = min_dumping_frames
        print(f">>> 正在加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)

    def expand_bbox_to_square(self, frame_shape, bbox):
        """
        将边界框扩展为正方形，同时保持中心点不变

        Args:
            frame_shape: 原始帧尺寸 (height, width)
            bbox: 原始边界框 [x1, y1, x2, y2]

        Returns:
            square_bbox: 正方形边界框 [x1, y1, x2, y2]
            adjusted: 是否进行了边界调整
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 目标边长取宽高的最大值
        target_size = max(bbox_w, bbox_h)

        # 计算新的边界
        new_x1 = center_x - target_size / 2
        new_x2 = center_x + target_size / 2
        new_y1 = center_y - target_size / 2
        new_y2 = center_y + target_size / 2

        # 边界调整标志
        adjusted = False

        # 处理超出图像边界的情况
        if new_x1 < 0:
            offset = -new_x1
            new_x1 = 0
            new_x2 = new_x2 + offset
            adjusted = True

        if new_y1 < 0:
            offset = -new_y1
            new_y1 = 0
            new_y2 = new_y2 + offset
            adjusted = True

        if new_x2 > w:
            offset = new_x2 - w
            new_x2 = w
            new_x1 = new_x1 - offset
            adjusted = True

        if new_y2 > h:
            offset = new_y2 - h
            new_y2 = h
            new_y1 = new_y1 - offset
            adjusted = True

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)], adjusted

    def extract_truck_region(self, frame, truck_box):
        """
        提取truck区域并扩展为正方形（不进行letterbox，直接resize）

        Args:
            frame: 原始帧图像
            truck_box: truck的边界框 [x1, y1, x2, y2]

        Returns:
            square_truck: 对齐后的正方形truck图片 (320x320)
        """
        x1, y1, x2, y2 = map(int, truck_box)

        # 扩展为正方形
        square_bbox, adjusted = self.expand_bbox_to_square(frame.shape, [x1, y1, x2, y2])
        x1, y1, x2, y2 = square_bbox

        # 确保坐标在帧范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # 裁剪truck区域
        truck_region = frame[y1:y2, x1:x2]

        if truck_region.size == 0:
            return None

        # 直接resize到320x320（不再使用letterbox填充）
        resized = cv2.resize(truck_region, (320, 320), interpolation=cv2.INTER_LINEAR)

        return resized

    def _check_horizontal_overlap_ratio(self, dumping_box, truck_box):
        """
        检查dumping和truck在x轴上的重叠比例

        Args:
            dumping_box: dumping的边界框 [x1, y1, x2, y2]
            truck_box: truck的边界框 [x1, y1, x2, y2]

        Returns:
            overlap_ratio: dumping被truck覆盖的比例（基于dumping的宽度）
        """
        dumping_x1, dumping_x2 = dumping_box[0], dumping_box[2]
        truck_x1, truck_x2 = truck_box[0], truck_box[2]

        # 计算重叠区间
        overlap_start = max(dumping_x1, truck_x1)
        overlap_end = min(dumping_x2, truck_x2)

        if overlap_start >= overlap_end:
            return 0.0

        overlap_width = overlap_end - overlap_start
        dumping_width = dumping_x2 - dumping_x1

        overlap_ratio = overlap_width / dumping_width if dumping_width > 0 else 0.0

        return overlap_ratio

    def _get_overlapping_trucks(self, dumping_box, truck_boxes, frame, min_overlap_ratio=0.5):
        """
        获取与dumping在x轴上重叠比例超过阈值的所有truck

        Args:
            dumping_box: dumping的边界框
            truck_boxes: 所有truck的边界框列表
            frame: 当前帧图像
            min_overlap_ratio: 最小重叠比例阈值（默认0.5）

        Returns:
            overlapping_trucks: 符合条件的truck列表
        """
        overlapping_trucks = []
        for truck in truck_boxes:
            overlap_ratio = self._check_horizontal_overlap_ratio(dumping_box, truck['xyxy'])
            if overlap_ratio >= min_overlap_ratio:
                truck_img = self.extract_truck_region(frame, truck['xyxy'])
                if truck_img is not None:
                    overlapping_trucks.append({
                        'track_id': truck['track_id'],
                        'image': truck_img,
                        'overlap_ratio': overlap_ratio
                    })
        return overlapping_trucks

    def _get_all_video_files(self, root_dir):
        """
        递归获取目录下所有视频文件

        Args:
            root_dir: 根目录路径

        Returns:
            video_files: 所有视频文件的路径列表
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.MP4', '.AVI'}
        video_files = []

        root_path = Path(root_dir)

        for file_path in root_path.rglob('*'):
            if file_path.suffix in video_extensions:
                video_files.append(file_path)

        return video_files

    def process_video(self, video_path, output_dir, global_counter):
        """
        处理单个视频，提取dumping相关的truck图片（仅post图片）
        """
        print(f"\n>>> 处理视频: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  警告: 无法打开视频 {video_path.name}，跳过")
            return 0

        # 状态机变量
        is_dumping_active = False
        pending_state = None
        state_confirm_count = 0

        # 当前dumping会话的数据
        current_session = {
            'start_frame': None,
            'end_frame': None,
            'end_trucks': [],
            'buffer': []
        }

        completed_sessions = []

        frame_count = 0
        session_count = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  总帧数: {total_frames}")

        start_time = time.time()
        last_log_time = start_time

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 进度显示
            current_time = time.time()
            if frame_count % 100 == 0 or (current_time - last_log_time) > 2:
                progress = (frame_count / total_frames) * 100
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  进度: {frame_count}/{total_frames} ({progress:.1f}%) - FPS: {fps:.1f}", end='\r')
                last_log_time = current_time

            try:
                results = self.model.track(frame, persist=True, tracker=self.tracker_config, verbose=False)

                has_dumping = False
                overlapping_trucks = []

                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    class_ids = boxes.cls.int().tolist()
                    track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(class_ids)
                    xyxy_list = boxes.xyxy.tolist()

                    truck_boxes = []
                    dumping_boxes = []

                    for i, (class_id, track_id, xyxy) in enumerate(zip(class_ids, track_ids, xyxy_list)):
                        if class_id == 2:
                            truck_boxes.append({'track_id': track_id, 'xyxy': xyxy})
                        elif class_id == 4:
                            dumping_boxes.append({'xyxy': xyxy, 'track_id': track_id})

                    if dumping_boxes:
                        has_dumping = True
                        best_dumping = dumping_boxes[0]
                        overlapping_trucks = self._get_overlapping_trucks(
                            best_dumping['xyxy'], truck_boxes, frame, min_overlap_ratio=0.5
                        )

                # 状态机逻辑
                current_state = has_dumping and len(overlapping_trucks) > 0

                if pending_state is None:
                    if current_state != is_dumping_active:
                        pending_state = current_state
                        state_confirm_count = 1
                        current_session['buffer'] = [{
                            'frame': frame_count,
                            'trucks': overlapping_trucks.copy(),
                            'frame_img': frame.copy()
                        }]
                    else:
                        if is_dumping_active:
                            current_session['end_frame'] = frame_count
                            current_session['end_trucks'] = overlapping_trucks.copy()
                else:
                    if current_state == pending_state:
                        state_confirm_count += 1
                        current_session['buffer'].append({
                            'frame': frame_count,
                            'trucks': overlapping_trucks.copy(),
                            'frame_img': frame.copy()
                        })

                        if state_confirm_count >= self.frame_buffer:
                            if pending_state == True:
                                is_dumping_active = True
                                session_count += 1
                                first_buffer = current_session['buffer'][0]
                                current_session['start_frame'] = first_buffer['frame']
                                current_session['end_frame'] = frame_count
                                current_session['end_trucks'] = overlapping_trucks.copy()

                                print(
                                    f"\n  ✓ [开始] Dumping #{session_count} 开始于帧 {current_session['start_frame']}")

                            else:
                                if is_dumping_active:
                                    session_duration = current_session['end_frame'] - current_session['start_frame']
                                    if session_duration >= self.min_dumping_frames:
                                        completed_sessions.append({
                                            'session_id': session_count,
                                            'end_frame': current_session['end_frame'],
                                            'end_trucks': current_session['end_trucks']
                                        })
                                        print(
                                            f"  ✓ [结束] Dumping #{session_count} 结束于帧 {current_session['end_frame']}, "
                                            f"持续 {session_duration} 帧, 结束重叠truck数: {len(current_session['end_trucks'])}")
                                    else:
                                        print(
                                            f"  ⚠ [过滤] Dumping #{session_count} 持续时间过短 ({session_duration}帧)，已过滤")

                                is_dumping_active = False

                            pending_state = None
                            state_confirm_count = 0
                            current_session['buffer'] = []
                    else:
                        pending_state = None
                        state_confirm_count = 0
                        current_session['buffer'] = []

            except Exception as e:
                print(f"\n  ⚠ 警告: 处理帧 {frame_count} 时出错: {e}")
                continue

        cap.release()

        print(f"\n  视频处理完成，共检测到 {len(completed_sessions)} 个有效的dumping会话")

        # ========== 只保存post图片 ==========
        saved_count = 0
        for session in completed_sessions:
            global_counter[0] += 1
            seq_num = f"{global_counter[0]:06d}"

            print(f"  💾 保存会话 #{seq_num}: 结束于帧 {session['end_frame']}")

            video_output_dir = output_dir / video_path.stem
            video_output_dir.mkdir(parents=True, exist_ok=True)

            # 只保存结束帧的trucks（post图片）
            for idx, truck_info in enumerate(session['end_trucks']):
                save_path = video_output_dir / f"{seq_num}_post_{idx}.jpg"
                cv2.imwrite(str(save_path), truck_info['image'])
                saved_count += 1
                print(f"    📸 保存: {save_path.name}")

        print(f"  ✅ 完成: 提取了 {len(completed_sessions)} 个dumping会话, 保存了 {saved_count} 张truck图片")

        return len(completed_sessions)

    def batch_process(self, input_dir, output_dir):
        """批量处理目录中的所有视频（支持递归搜索子目录）"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 递归获取所有视频文件
        video_files = self._get_all_video_files(input_dir)

        if not video_files:
            print(f"错误: 在 {input_dir} 及其子目录中未找到视频文件")
            return

        print(f"\n找到 {len(video_files)} 个视频文件")
        print(f"输出目录: {output_path}")

        global_counter = [0]
        total_sessions = 0

        for idx, video_file in enumerate(video_files, 1):
            print(f"\n{'=' * 60}")
            print(f"处理第 {idx}/{len(video_files)} 个视频")
            print(f"视频路径: {video_file}")
            print(f"{'=' * 60}")

            try:
                sessions = self.process_video(video_file, output_path, global_counter)
                total_sessions += sessions
            except Exception as e:
                print(f"  ❌ 错误: 处理视频 {video_file.name} 时出错: {e}")
                traceback.print_exc()
                continue

        print(f"\n{'=' * 60}")
        print(f"批量处理完成!")
        print(f"总共处理视频数: {len(video_files)}")
        print(f"总共提取dumping会话数: {total_sessions}")
        print(f"总共保存post图片数: {global_counter[0]}")
        print(f"输出目录: {output_path}")
        print(f"{'=' * 60}")


def main():
    # 配置参数
    MODEL_PATH = r"E:\pycharmProjects\Sany-Excavator-Count\infer\best.pt"
    INPUT_VIDEO_DIR = r"E:\data\mp4"  # 会递归搜索所有子目录
    OUTPUT_DIR = r"E:\pycharmProjects\Sany-Excavator-Count\infer\dumping_truck_extractsV1.1"

    extractor = DumpingTruckExtractor(
        model_path=MODEL_PATH,
        tracker_config="bytetrack.yaml",
        frame_buffer=5,
        min_dumping_frames=10
    )

    extractor.batch_process(INPUT_VIDEO_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()