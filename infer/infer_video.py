import sys
import io
import numpy as np
import time
import torch

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


class VideoTracker:
    def __init__(self, video_path, model_path, output_path, siamese_model_path, tracker_config="bytetrack.yaml", callback=None):
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

        self.callback = callback

        # 计数器和状态标志
        self.total_truck_count = 0
        self.total_bucket_count = 0
        self.bucket_full = False
        self.dumping_active = False

        self.current_ticket_id = None  # 当前装车票号
        self.reference_truck_img = None  # 当前票号对应的车辆特征底图（锁定不漂移）

        # 加载模型
        print(f">>> 正在加载模型: {self.model_path}")
        self.model = YOLO(self.model_path)

        # 加载孪生网络模型（用于车辆重识别）
        print(f">>> 正在加载孪生网络模型...")
        self.siamese_model = None
        self.siamese_transform = None
        self.init_siamese_model(siamese_model_path)

        # 视频相关属性（在 run_video_inference 中初始化）
        self.cap = None
        self.out = None
        self.width = None
        self.height = None
        self.fps = None
        self.total_frames = None

    def init_siamese_model(self, siamese_model_path):
        """
        初始化孪生网络模型

        Args:
            siamese_model_path: 训练好的孪生模型权重路径
        """
        # 导入必要的模块
        from pathlib import Path
        import sys

        # 添加模型路径（根据你的实际目录调整）
        model_path = Path(siamese_model_path).parent
        if str(model_path) not in sys.path:
            sys.path.insert(0, str(model_path))

        # 导入孪生网络模型
        try:
            from siamese_model.siamese_network import load_siamese_model
            from utils.transforms import get_transforms

            self.siamese_model, _ = load_siamese_model(siamese_model_path, self.device)
            self.siamese_model.eval()
            self.siamese_transform = get_transforms(224, is_train=False)  # 224是输入尺寸

            print(f">>> 孪生网络模型加载成功，设备: {self.device}")

        except ImportError as e:
            print(f"⚠️ 无法加载孪生网络模块: {e}")
            print("将跳过车辆重识别功能")
            self.siamese_model = None

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
    def _generate_ticket_id():
        """生成唯一的装车票号 (使用时间戳)"""
        import time
        return f"TKT_{int(time.time() * 1000)}"

    @staticmethod
    def _letterbox(image, target_size=(224, 224)):
        """
        使用letterbox方法将图片对齐到目标尺寸，保持宽高比

        Args:
            image: 输入图片 (numpy array)
            target_size: 目标尺寸 (height, width)

        Returns:
            letterboxed_image: 对齐后的图片
        """
        if isinstance(target_size, tuple):
            target_h, target_w = target_size
        else:
            target_h = target_w = target_size

        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 计算填充
        dw = target_w - new_w
        dh = target_h - new_h
        pad_left = dw // 2
        pad_right = dw - pad_left
        pad_top = dh // 2
        pad_bottom = dh - pad_top

        # 填充
        letterboxed = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return letterboxed

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

        优化策略：
        1. bucket：只有与 truck 有横坐标重叠时，才进行计数逻辑
        2. truck：引入状态变化计数器，连续多帧确认后才执行状态转换

        Args:
            yolo_results: YOLO 推理结果（Results对象）

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
        has_dumping = len(dumping_boxes) > 0

        # 原版核心防抖：如果倒土时铲斗不是满的（意味着土已经倒下去了），直接 return，等待动作结束
        if has_dumping and not self.bucket_full:
            return

        if has_dumping and not self.dumping_active:
            self.dumping_active = True

        # 记录最后一帧的 dumping 位置（有 dumping 时持续更新）
        if has_dumping:
            self._last_dumping_box = dumping_boxes[0]['xyxy']

        # 原版神级逻辑：只有当 dumping 彻底结束（扬尘消散），且土倒下去后，才截图比对
        if not has_dumping and self.dumping_active and not self.bucket_full:

            if self._last_dumping_box is not None and truck_boxes:
                dumping_center_x = (self._last_dumping_box[0] + self._last_dumping_box[2]) / 2

                closest_truck = None
                min_distance = float('inf')

                for truck in truck_boxes:
                    if self._check_horizontal_overlap(self._last_dumping_box, truck['xyxy']):
                        truck_center_x = (truck['xyxy'][0] + truck['xyxy'][2]) / 2
                        distance = abs(truck_center_x - dumping_center_x)

                        if distance < min_distance:
                            min_distance = distance
                            closest_truck = truck

                if closest_truck:
                    # 扩展边界框并提取truck图片
                    expanded_bbox = self._expand_bbox_for_truck(closest_truck['xyxy'], frame.shape)
                    x1, y1, x2, y2 = expanded_bbox
                    truck_img = frame[y1:y2, x1:x2]

                    if truck_img.size > 0:
                        current_truck_img = self._letterbox(truck_img, (224, 224))

                        # --- 修复原版缺失的第一辆车判断，并结合票号系统 ---
                        if self.reference_truck_img is None:
                            # 刚开机，第一辆车倒完第一铲！立刻生成票号并 +1
                            self.current_ticket_id = self._generate_ticket_id()
                            self.reference_truck_img = current_truck_img
                            self.total_truck_count += 1
                            print(
                                f"  [计数] 第一辆车入场并完成首铲，票号: {self.current_ticket_id}，总数: {self.total_truck_count}")

                        else:
                            # 老规矩，交给 MobileNet 孪生网络判断
                            result = self._compare_trucks(self.reference_truck_img, current_truck_img)

                            if result.get('is_same') is False:
                                # 换新车了，且新车的第一铲土刚倒完！立刻生成新票号并 +1
                                self.current_ticket_id = self._generate_ticket_id()
                                self.reference_truck_img = current_truck_img
                                self.total_truck_count += 1
                                print(
                                    f"  [计数] 换车！新车完成首铲，票号: {self.current_ticket_id}，总数: {self.total_truck_count}")
                            else:
                                # 还是这辆车，滚动更新一下参考图，防止特征漂移
                                self.reference_truck_img = current_truck_img

            # 状态机彻底重置，等待下一次挖机倒土
            self.dumping_active = False

    def _compare_trucks(self, img1, img2, threshold=0.75):
        """
        使用孪生网络比较两张truck图片是否为同一辆车

        Args:
            img1: numpy array格式的第一张图片
            img2: numpy array格式的第二张图片
            threshold: 判断阈值

        Returns:
            dict: {'is_same': bool, 'similarity': float}
        """
        if self.siamese_model is None:
            # 没有孪生模型时，默认判断为同一辆车（不计数）
            return {'is_same': True, 'similarity': 1.0}

        # 转换numpy为tensor
        img1_tensor = self._numpy_to_tensor(img1)
        img2_tensor = self._numpy_to_tensor(img2)

        # 推理
        with torch.no_grad():
            similarity = self.siamese_model.compare(img1_tensor, img2_tensor)
            similarity = similarity.item()

        is_same = similarity > threshold

        return {'is_same': is_same, 'similarity': similarity}

    def _numpy_to_tensor(self, img):
        """
        将numpy格式的图片转换为模型输入tensor

        Args:
            img: numpy array, shape (H, W, C), BGR顺序

        Returns:
            tensor: shape (1, 3, 224, 224), RGB顺序, 归一化
        """
        from PIL import Image

        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 将numpy array转换为PIL Image
        img_pil = Image.fromarray(img_rgb)

        # 应用transform
        tensor = self.siamese_transform(img_pil)

        # 添加batch维度
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

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

        # 计算新画布尺寸
        display_width = self.height // 2  # 显示区域宽度 = 视频高度的一半（保持正方形）
        new_width = self.width + display_width + 20  # 原宽度 + 显示区宽度 + 边距
        new_height = self.height

        # 创建视频写入器（使用新尺寸）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (new_width, new_height))

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
            self._update_state_machine(results[0], frame)
            trucks, buckets = self.get_counts()

            if self.callback:
                # 只有当有了票号才发送有效数据
                if self.current_ticket_id is not None and not self.dumping_active:
                    self.callback({
                        "ticket_id": self.current_ticket_id,
                        "truck_count": trucks,
                        "bucket_count": buckets,
                        "timestamp": time.time(),
                    })

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

            # 在渲染部分，创建画布
            output_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            # 原视频放在左侧
            output_canvas[0:self.height, 0:self.width] = annotated_frame

            # 右侧显示区域
            right_x = self.width + 10
            display_block_height = self.height // 2 - 10  # 每个方块高度（留边距）
            display_block_width = display_block_height  # 正方形

            # 显示 reference_truck_img (当前票号锁定的车辆底图)
            if hasattr(self, 'reference_truck_img') and self.reference_truck_img is not None:
                # 调整底图大小并放到右上角
                display_img = cv2.resize(self.reference_truck_img, (display_block_width, display_block_height))
                output_canvas[10:10 + display_block_height, right_x:right_x + display_block_width] = display_img

                # 绘制文字：标识这是参考底图
                cv2.putText(output_canvas, "Locked Target", (right_x + 5, 30), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                # 顺便把当前的 Ticket ID 也显示在下方！
                if self.current_ticket_id:
                    cv2.putText(output_canvas, f"ID: {self.current_ticket_id}",
                                (right_x, 10 + display_block_height + 30),
                                font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                # 如果还没车来，画个灰色的占位框
                cv2.rectangle(output_canvas, (right_x, 10), (right_x + display_block_width, 10 + display_block_height),
                              (50, 50, 50), -1)
                cv2.putText(output_canvas, "Waiting for truck...", (right_x + 5, 30), font, 0.5, (200, 200, 200), 1,
                            cv2.LINE_AA)

            # 写入画布
            self.out.write(output_canvas)

            # 打印进度提示
            if frame_count % 50 == 0:
                print(f"进度: {frame_count} / {self.total_frames} 帧...")

        # 释放资源
        self.cap.release()
        self.out.release()
        print(f"\n>>> 推断完成！输出视频已保存至: {self.output_path}")


if __name__ == "__main__":
    TEST_VIDEO = "./tmp_files/test_video_shift.mp4"
    TRAINED_MODEL = "./tmp_files/best.pt"
    OUTPUT_VIDEO = "./tmp_files/output.mp4"
    SIAMESE_MODEL_PATH = "./tmp_files/attention_siamese_best.pth"

    # 模拟外部业务接收函数
    def client_business_logic(data):
        print(f"回调数据: {data}")

    # 创建跟踪器实例，传入 callback
    tracker = VideoTracker(
        video_path=TEST_VIDEO,
        model_path=TRAINED_MODEL,
        output_path=OUTPUT_VIDEO,
        siamese_model_path=SIAMESE_MODEL_PATH,
        tracker_config="bytetrack.yaml",
        callback=client_business_logic,
    )

    # 运行推理
    tracker.run_video_inference()
