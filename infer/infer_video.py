import sys
import io

# 强制将标准输出和错误输出设置为 utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
from ultralytics import YOLO


def update_state_machine(yolo_results):
    """
    处理 YOLO 当前帧的输出，更新状态机逻辑。
    """
    pass

    # 暂时返回固定的测试数字，用于 UI 渲染
    dummy_truck_count = 0
    dummy_bucket_count = 0
    return dummy_truck_count, dummy_bucket_count

def run_video_inference(video_path, model_path, output_path):
    print(f">>> 正在加载模型: {model_path}")
    model = YOLO(model_path)

    print(f">>> 正在打开视频: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频，请检查路径！")

    # 获取原视频的宽度、高度和帧率，用于保存输出视频
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化视频写入器 (mp4v 编码兼容性较好)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    print(">>> 开始逐帧推断并渲染...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---------------------------------------------------------
        # A. 运行 YOLO 推断与目标跟踪
        # persist=True 告诉追踪器记住上一帧的对象，维持 ID
        # verbose=False 关闭控制台疯狂刷新的单帧日志
        # ---------------------------------------------------------
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        # ---------------------------------------------------------
        # B. 调用状态机 (目前为空，返回 0)
        # ---------------------------------------------------------
        trucks, buckets = update_state_machine(results[0])

        # ---------------------------------------------------------
        # C. 画面渲染
        # ---------------------------------------------------------
        # 1. 直接让 YOLO 帮我们把 Bounding Box 和 ID 画到画面上
        annotated_frame = results[0].plot()

        # 2. 在右上角添加 UI 计数器面板
        # 背景底色 (为了让文字在灰尘和高光下也清晰可见)
        ui_x1, ui_y1 = width - 350, 20
        ui_x2, ui_y2 = width - 20, 120
        cv2.rectangle(annotated_frame, (ui_x1, ui_y1), (ui_x2, ui_y2), (0, 0, 0), -1)

        # 绘制文本 (绿色字)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, f"Trucks:  {trucks}", (width - 330, 60), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Buckets: {buckets}", (width - 330, 105), font, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        # 写入文件
        out.write(annotated_frame)

        # 打印进度提示
        if frame_count % 50 == 0:
            print(f"进度: {frame_count} / {total_frames} 帧...")

    # 释放资源
    cap.release()
    out.release()
    print(f"\n>>> 推断完成！输出视频已保存至: {output_path}")


if __name__ == "__main__":
    TEST_VIDEO = "./JFSK_20251230_110914_N1_00.mp4"  # 输入的测试视频

    # TRAINED_MODEL = "Excavator_Counting/yolov8n_edge_v12/weights/best.pt"  # 训练出的最佳权重
    TRAINED_MODEL = "./best.pt"

    OUTPUT_VIDEO = "output_inference.mp4"  # 输出的视频名

    run_video_inference(TEST_VIDEO, TRAINED_MODEL, OUTPUT_VIDEO)