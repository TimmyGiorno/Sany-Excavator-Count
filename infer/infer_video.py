import sys
import io

import numpy as np
import torch
from PIL import Image

# 强制将标准输出和错误输出设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
from ultralytics import YOLO

# ========== 强制实时输出 ==========
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import functools

print = functools.partial(print, flush=True)


# =================================


class VideoTracker:
    def __init__(self, video_path, model_path, output_path, siamese_model_path, tracker_config="bytetrack.yaml"):
        """
        初始化视频跟踪器

        Args:
            video_path: 输入视频路径
            model_path: 模型权重路径
            output_path: 输出视频路径
            tracker_config: 跟踪器配置文件路径
        """
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.tracker_config = tracker_config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 计数器和状态标志
        self.total_truck_count = 0
        self.total_bucket_count = 0
        self.bucket_full = False
        self.dumping_active = False

        # 统计相关
        self.stat_frames_remaining = 0  # 还需要统计多少帧
        self.ratio_buffer = []  # 存储每帧的 mine/truck 面积比值
        self.last_avg_ratio = None  # 上一次 dumping 后的平均比值
        self.is_statting = False  # 是否正在统计中
        self.stat_frame_count = 0  # 已统计的帧数
        self._last_dumping_box = None
        self._last_dumping_bucket_xyxy = None
        self.dumping_frame_count = 0  # dumping 连续出现的帧数计数

        # 稳定期相关
        self.stable_frames_remaining = 0  # 还需要等待多少帧稳定期

        # 当前作业 truck 位置（每帧重新计算）
        self.current_truck_xyxy = None  # 当前正在作业的 truck 的边界框

        # 统计周期（连续统计多少帧）
        self.stat_window_frames = 10

        # 下降阈值（75%）
        self.decline_threshold = 0.75

        # 加载模型
        print(f">>> 正在加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)

        # 视频相关属性（在 run_video_inference 中初始化）
        self.cap = None
        self.out = None
        self.width = None
        self.height = None
        self.fps = None
        self.total_frames = None

    def get_counts(self):
        """
        获取当前的累计计数

        Returns:
            tuple: (total_truck_count, total_bucket_count)
        """
        return self.total_truck_count, self.total_bucket_count

    def reset_counts(self):
        """
        重置计数器和状态标志
        """
        self.total_truck_count = 0
        self.total_bucket_count = 0
        self.bucket_full = False
        print(">>> 计数器已重置")

    @staticmethod
    def _expand_bbox_for_truck(bbox, frame_shape):
        """
        根据卡车框的形状进行扩展：
        - 如果宽度 > 高度：向上延长高度（纳入上方矿物）
        - 如果高度 > 宽度：向左右平均延长
        - 保持底边/中心不变（根据需求调整）

        Args:
            bbox: [x1, y1, x2, y2] (y1是上边界，y2是下边界)
            frame_shape: (height, width)

        Returns:
            expanded_bbox: [x1, y1, x2, y2]
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        if width > height:
            # 宽度大于高度：向上延长高度，保持底边不变
            new_height = width  # 目标高度 = 当前宽度
            new_y1 = max(0, y2 - new_height)  # 向上延长，保持y2不变
            new_y2 = y2
            new_x1 = x1
            new_x2 = x2
        else:
            # 高度大于宽度：向左右平均延长
            new_width = height  # 目标宽度 = 当前高度
            total_dx = new_width - width
            left_dx = total_dx // 2
            right_dx = total_dx - left_dx
            new_x1 = max(0, x1 - left_dx)
            new_x2 = min(w, x2 + right_dx)
            new_y1 = y1
            new_y2 = y2

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

    def _update_state_machine(self, yolo_results, frame):
        """
        处理 YOLO 当前帧的输出，更新状态机逻辑。

        Args:
            yolo_results: YOLO 推理结果（Results对象）
            frame: 当前帧图像（用于面积计算）

        """
        if yolo_results.boxes is None:
            return

        # 获取所有检测结果
        boxes = yolo_results.boxes
        class_ids = boxes.cls.int().tolist()
        track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(class_ids)
        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()

        # 准备数据容器
        truck_boxes = []  # 存储所有 truck (class_id=2)
        bucket_boxes = []  # 存储所有 bucket (class_id=0,1)
        dumping_boxes = []  # 存储所有 dumping (class_id=4)
        mine_boxes = []  # 存储所有 mine (class_id=5)

        # 分类存储检测结果
        for i, (class_id, track_id, xyxy, conf) in enumerate(zip(class_ids, track_ids, xyxy_list, conf_list)):
            if class_id == 2:  # truck
                truck_boxes.append({
                    'class_id': class_id,
                    'track_id': track_id,
                    'xyxy': xyxy,
                    'conf': conf,
                    'class_name': 'truck'
                })
            elif class_id == 5:  # mine
                mine_boxes.append({
                    'class_id': class_id,
                    'track_id': track_id,
                    'xyxy': xyxy,
                    'conf': conf,
                    'class_name': 'mine'
                })
            elif class_id in [0, 1]:  # bucket-empty 或 bucket-full
                bucket_boxes.append({
                    'class_id': class_id,
                    'track_id': track_id,
                    'xyxy': xyxy,
                    'conf': conf,
                    'class_name': 'bucket-full' if class_id == 1 else 'bucket-empty'
                })
            elif class_id == 4:  # dumping
                dumping_boxes.append({
                    'class_id': class_id,
                    'track_id': track_id,
                    'xyxy': xyxy,
                    'conf': conf,
                    'class_name': 'dumping'
                })

        # ============================================================
        # 1. 处理 bucket 状态转换（需要与 truck 有重叠才计数）
        # ============================================================
        if bucket_boxes:
            # 找出置信度最高的 bucket
            best_bucket = max(bucket_boxes, key=lambda x: x['conf'])
            class_id = best_bucket['class_id']

            # bucket-full 检测
            if class_id == 1:  # bucket-full
                if not self.bucket_full:
                    # 检查是否与任何 truck 有横坐标重叠
                    has_overlap = False
                    for truck in truck_boxes:
                        if self._check_horizontal_overlap(best_bucket['xyxy'], truck['xyxy']):
                            has_overlap = True
                            break

                    # 只有当挖斗与矿车无重叠时，才允许变为 full
                    if not has_overlap:
                        self.bucket_full = True

            # bucket-empty 检测
            elif class_id == 0:  # bucket-empty
                if self.bucket_full:
                    # 检查是否与任何 truck 有横坐标重叠
                    has_overlap = False
                    for truck in truck_boxes:
                        if self._check_horizontal_overlap(best_bucket['xyxy'], truck['xyxy']):
                            has_overlap = True
                            break

                    if has_overlap:
                        # 有重叠：完成计数，重置状态
                        self.bucket_full = False
                        self.total_bucket_count += 1

                        # bucket 完成倾倒后，开始等待 dumping 出现
                        # （dumping 会在 bucket empty 之后出现）
                    else:
                        # 无重叠：只重置状态，不计数
                        self.bucket_full = False

        # ============================================================
        # 2. 处理 dumping 状态跟踪
        # ============================================================
        has_dumping = len(dumping_boxes) > 0

        if has_dumping:
            # 每帧都更新 dumping 位置
            self._last_dumping_box = dumping_boxes[0]['xyxy']

            # 累计 dumping 连续帧数
            self.dumping_frame_count += 1

            if not self.dumping_active:
                # 只有达到 5 帧以上才标记为真正的 dumping 开始
                if self.dumping_frame_count >= 5:
                    self.dumping_active = True

            # 如果当前处于稳定期或统计期，说明之前的 dumping 状态不稳定，需要放弃
            if self.stable_frames_remaining > 0 or self.is_statting:
                self.stable_frames_remaining = 0
                self.is_statting = False
                self.stat_frames_remaining = 0
                self.ratio_buffer = []
                self.current_truck_xyxy = None
                self._last_dumping_bucket_xyxy = None

        elif not has_dumping:
            # 当前帧没有 dumping
            if self.dumping_frame_count > 0:
                # 重置连续帧计数
                self.dumping_frame_count = 0

            if self.dumping_active:
                # 真正的 dumping 结束，开始稳定期
                self.dumping_active = False
                self.stable_frames_remaining = 1

                # 记录用于找 truck 的 bucket 位置
                if bucket_boxes:
                    best_bucket = max(bucket_boxes, key=lambda x: x['conf'])
                    self._last_dumping_bucket_xyxy = best_bucket['xyxy']
                else:
                    self._last_dumping_bucket_xyxy = None

        # ============================================================
        # 3. 处理稳定期
        # ============================================================
        if self.stable_frames_remaining > 0:
            self.stable_frames_remaining -= 1

            if self.stable_frames_remaining == 0:
                # 稳定期结束，准备开始统计

                # 如果没有 bucket 位置记录，尝试从当前帧获取
                if self._last_dumping_bucket_xyxy is None and bucket_boxes:
                    best_bucket = max(bucket_boxes, key=lambda x: x['conf'])
                    self._last_dumping_bucket_xyxy = best_bucket['xyxy']
                    print("  [补充] 从当前帧获取 bucket 位置")

                if self._last_dumping_bucket_xyxy is not None:
                    if truck_boxes:
                        # 找到 truck，开始统计
                        self.is_statting = True
                        self.ratio_buffer = []
                        self.stat_frames_remaining = 10

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
                        # 没有 truck，延迟一帧再试
                        self.stable_frames_remaining = 1
                        print("  [等待] 稳定期结束但未找到作业中的 truck，延迟一帧")
                else:
                    # 没有 bucket 位置记录，且当前帧也没有 bucket，延迟一帧
                    self.stable_frames_remaining = 1
                    print("  [等待] 稳定期结束但没有 bucket 位置记录，且当前帧无 bucket，延迟一帧")

        # ============================================================
        # 4. 统计 mine/truck 面积比值
        # ============================================================
        if self.is_statting and self.stat_frames_remaining > 0:
            self.stat_frames_remaining -= 1
            self.stat_frame_count += 1

            # 如果有正在作业的 truck，找这一帧中离它最近的 truck
            if self.current_truck_xyxy is not None and truck_boxes:
                # 找与上一帧 truck 位置最近的 truck
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
                    # 更新当前作业 truck 的位置
                    self.current_truck_xyxy = current_truck['xyxy']

                    # 计算 truck 面积
                    truck_x1, truck_y1, truck_x2, truck_y2 = self.current_truck_xyxy
                    truck_area = (truck_x2 - truck_x1) * (truck_y2 - truck_y1)

                    if truck_area > 0:
                        # 找与 truck 有水平交集的 mine 中最近的那个
                        mine_area = 0
                        truck_center_x = (truck_x1 + truck_x2) / 2

                        for mine in mine_boxes:
                            # 检查水平方向是否有重叠
                            if self._check_horizontal_overlap(self.current_truck_xyxy, mine['xyxy']):
                                # 计算 mine 面积
                                mine_x1, mine_y1, mine_x2, mine_y2 = mine['xyxy']
                                current_mine_area = (mine_x2 - mine_x1) * (mine_y2 - mine_y1)

                                # 如果有多个 mine，取离 truck 中心最近的
                                mine_center_x = (mine_x1 + mine_x2) / 2
                                distance = abs(mine_center_x - truck_center_x)

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

            # 统计完成
            if self.stat_frames_remaining == 0:
                self.is_statting = False

                # 计算平均比值
                avg_ratio = sum(self.ratio_buffer) / len(self.ratio_buffer) if self.ratio_buffer else 0.0

                print(f"  [统计] 统计了 {len(self.ratio_buffer)} 帧，平均 mine/truck 比值: {avg_ratio:.4f}")

                # 判断是否需要计数
                if self.last_avg_ratio is None:
                    # 第一次，只记录，不计数
                    self.last_avg_ratio = avg_ratio
                    print(f"  [统计] 第一次记录基准值: {avg_ratio:.4f}")
                else:
                    # 特殊情况：上一次和本次都是 0
                    if self.last_avg_ratio == 0.0 and avg_ratio == 0.0:
                        # 都是空车，视为同一辆车，不计数
                        print(f"  [统计] 两次均为空车，视为同一辆车，不计数")
                    else:
                        # 计算下降比例
                        if self.last_avg_ratio > 0:
                            decline = (self.last_avg_ratio - avg_ratio) / self.last_avg_ratio
                        else:
                            # 上一次是 0，本次 > 0，说明从空车变成有矿，下降比例无意义，不计数
                            decline = 0.0

                        print(
                            f"  [统计] 上一次平均值: {self.last_avg_ratio:.4f}, 本次平均值: {avg_ratio:.4f}, 下降: {decline * 100:.1f}%")

                        if avg_ratio == 0.0 or decline >= 0.75:
                            self.total_truck_count += 1
                            print(f"  [计数] 发现新作业车辆，truck 计数 +1，当前总数: {self.total_truck_count}")

                    # 更新基准值为本次平均值
                    self.last_avg_ratio = avg_ratio

                # 清理状态
                self.current_truck_xyxy = None
                self._last_dumping_bucket_xyxy = None

    @staticmethod
    def _check_horizontal_overlap(box1, box2):
        """
        检查两个边界框在水平方向（x轴）是否有重叠

        Args:
            box1: [x1, y1, x2, y2] 第一个框的坐标
            box2: [x1, y1, x2, y2] 第二个框的坐标

        Returns:
            bool: 水平方向有重叠返回 True，否则 False
        """
        # 获取两个框的 x 坐标范围
        x1_min, x1_max = box1[0], box1[2]
        x2_min, x2_max = box2[0], box2[2]

        # 检查是否有重叠（区间重叠检测）
        overlap = not (x1_max < x2_min or x2_max < x1_min)

        return overlap

    def run_video_inference(self):
        """
        执行视频推断的主方法
        """
        print(f">>> 正在打开视频: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频，请检查路径！")

        # 获取原视频的宽度、高度和帧率，用于保存输出视频
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入器（使用原视频尺寸）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0

        print(">>> 开始逐帧推断并渲染...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            # ---------------------------------------------------------
            # A. 运行 YOLO 推断与目标跟踪
            # persist=True 告诉追踪器记住上一帧的对象，维持 ID
            # verbose=False 关闭控制台疯狂刷新的单帧日志
            # ---------------------------------------------------------
            results = self.model.track(frame, persist=True, tracker=self.tracker_config, verbose=False)

            # ---------------------------------------------------------
            # B. 调用状态机
            # ---------------------------------------------------------
            self._update_state_machine(results[0], frame)
            trucks, buckets = self.get_counts()

            # ---------------------------------------------------------
            # C. 画面渲染
            # ---------------------------------------------------------
            # 1. 让 YOLO 把 Bounding Box 和 ID 画到画面上
            annotated_frame = results[0].plot()

            # 2. 在右上角添加 UI 计数器面板
            ui_x1, ui_y1 = self.width - 350, 20
            ui_x2, ui_y2 = self.width - 20, 120
            cv2.rectangle(annotated_frame, (ui_x1, ui_y1), (ui_x2, ui_y2), (0, 0, 0), -1)

            # 绘制文本 (绿色字)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_frame, f"Trucks:  {trucks}", (self.width - 330, 60), font, 1.2, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Buckets: {buckets}", (self.width - 330, 105), font, 1.2, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # 直接写入原视频画面
            self.out.write(annotated_frame)

            # 打印进度提示
            if frame_count % 50 == 0:
                print(f"进度: {frame_count} / {self.total_frames} 帧...")

        # 释放资源
        self.cap.release()
        self.out.release()
        print(f"\n>>> 推断完成！输出视频已保存至: {self.output_path}")


if __name__ == "__main__":
    # TEST_VIDEO = "./JFSK_20251230_165914_N1_00.mp4"  # 输入的测试视频
    # TEST_VIDEO = "E:/data/mp4/JFSK_20251230_140914_N1_00.mp4"
    TEST_VIDEO = "./9f8cedc79d8b824a95cca11894ca232a.mp4"
    TRAINED_MODEL = "./best.pt"  # 训练出的最佳权重
    OUTPUT_VIDEO = "test5.mp4"  # 输出的视频名

    # 创建跟踪器实例
    tracker = VideoTracker(
        video_path=TEST_VIDEO,
        model_path=TRAINED_MODEL,
        output_path=OUTPUT_VIDEO,
        tracker_config="bytetrack.yaml"
    )

    # 运行推理
    tracker.run_video_inference()

    # 示例：获取计数和重置
    # truck_count, bucket_count = tracker.get_counts()
    # print(f"最终统计 - 卡车: {truck_count}, 铲斗: {bucket_count}")
    # tracker.reset_counts()
