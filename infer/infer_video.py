import sys
import io

# 强制将标准输出和错误输出设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
from ultralytics import YOLO


class VideoTracker:
    def __init__(self, video_path, model_path, output_path, tracker_config="bytetrack.yaml"):
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

        # 计数器和状态标志
        self.total_truck_count = 0
        self.total_bucket_count = 0
        self.truck_full = False
        self.bucket_full = False
        self.truck_state_change_count = 0
        self.state_change_threshold = 25

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
        self.truck_full = False
        self.bucket_full = False
        self.truck_state_change_count = 0
        print(">>> 计数器已重置")

    def _update_state_machine(self, yolo_results):
        """
        处理 YOLO 当前帧的输出，更新状态机逻辑。

        优化策略：
        1. bucket：只有与 truck 有横坐标重叠时，才进行计数逻辑
        2. truck：引入状态变化计数器，连续多帧确认后才执行状态转换

        Args:
            yolo_results: YOLO 推理结果（Results对象）

        """
        if yolo_results.boxes is None:
            return self.total_truck_count, self.total_bucket_count

        # 获取所有检测结果
        boxes = yolo_results.boxes
        class_ids = boxes.cls.int().tolist()
        track_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(class_ids)
        xyxy_list = boxes.xyxy.tolist()  # 获取所有边界框坐标 [x1, y1, x2, y2]
        conf_list = boxes.conf.tolist()

        # 准备数据容器
        truck_boxes = []  # 存储所有 truck 的边界框
        bucket_boxes = []  # 存储所有 bucket 的边界框

        # 分类存储检测结果
        for i, (class_id, track_id, xyxy, conf) in enumerate(zip(class_ids, track_ids, xyxy_list, conf_list)):
            if class_id in [2, 3]:  # truck-empty 或 truck-full
                truck_boxes.append({
                    'class_id': class_id,
                    'track_id': track_id,
                    'xyxy': xyxy,
                    'conf': conf,
                    'class_name': 'truck-full' if class_id == 3 else 'truck-empty'
                })
            elif class_id in [0, 1]:  # bucket-empty 或 bucket-full
                bucket_boxes.append({
                    'class_id': class_id,
                    'track_id': track_id,
                    'xyxy': xyxy,
                    'conf': conf,
                    'class_name': 'bucket-full' if class_id == 1 else 'bucket-empty'
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

                    # 只有当挖斗与矿车无重叠时，才允许变为 full（表示挖斗已离开矿车去挖矿）
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
                    else:
                        # 无重叠：只重置状态，不计数
                        self.bucket_full = False

        # ============================================================
        # 2. 处理 truck 状态转换（需要连续多帧确认）
        # ============================================================

        # TODO:对于一帧内有多辆卡车的情况进行处理

        for truck in truck_boxes:
            class_id = truck['class_id']

            if class_id == 3:  # truck-full
                if not self.truck_full:
                    self.truck_state_change_count += 1
                    if self.truck_state_change_count >= self.state_change_threshold:
                        self.truck_state_change_count = 0
                        self.truck_full = True
                else:
                    self.truck_state_change_count = 0

            elif class_id == 2:  # truck-empty
                if self.truck_full:
                    self.truck_state_change_count += 1
                    if self.truck_state_change_count >= self.state_change_threshold:
                        self.truck_state_change_count = 0
                        self.truck_full = False
                        self.total_truck_count += 1
                else:
                    self.truck_state_change_count = 0

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

        # 初始化视频写入器 (mp4v 编码兼容性较好)
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
            # B. 调用状态机 (目前返回当前累计值)
            # ---------------------------------------------------------
            self._update_state_machine(results[0])
            trucks, buckets = self.get_counts()

            # ---------------------------------------------------------
            # C. 画面渲染
            # ---------------------------------------------------------
            # 1. 直接让 YOLO 帮我们把 Bounding Box 和 ID 画到画面上
            annotated_frame = results[0].plot()

            # 2. 在右上角添加 UI 计数器面板
            # 背景底色 (为了让文字在灰尘和高光下也清晰可见)
            ui_x1, ui_y1 = self.width - 350, 20
            ui_x2, ui_y2 = self.width - 20, 120
            cv2.rectangle(annotated_frame, (ui_x1, ui_y1), (ui_x2, ui_y2), (0, 0, 0), -1)

            # 绘制文本 (绿色字)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated_frame, f"Trucks:  {trucks}", (self.width - 330, 60), font, 1.2, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Buckets: {buckets}", (self.width - 330, 105), font, 1.2, (0, 255, 0), 2,
                        cv2.LINE_AA)

            # 写入文件
            self.out.write(annotated_frame)

            # 打印进度提示
            if frame_count % 50 == 0:
                print(f"进度: {frame_count} / {self.total_frames} 帧...")

        # 释放资源
        self.cap.release()
        self.out.release()
        print(f"\n>>> 推断完成！输出视频已保存至: {self.output_path}")


if __name__ == "__main__":
    TEST_VIDEO = "./JFSK_20251230_165914_N1_00.mp4"  # 输入的测试视频
    TRAINED_MODEL = "./best.pt"  # 训练出的最佳权重
    OUTPUT_VIDEO = "output_inference.mp4"  # 输出的视频名

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
