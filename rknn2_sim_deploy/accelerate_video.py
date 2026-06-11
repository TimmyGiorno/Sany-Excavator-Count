import cv2
import os
import time
from typing import Dict, Any

# ================= 配置区 ============================================================
VIDEO_CONFIG = {
    'INPUT_VIDEO': './tmp_files/test_video_shift.mp4',  # 原始视频路径
    'OUTPUT_VIDEO': './tmp_files/test_video_shift_fast.mp4',  # 加速后的视频保存路径
    'SPEED_FACTOR': 5.0,  # 加速倍数（例如：4.0 表示 4 倍速）
}

# ====================================================================================
def accelerate_video(config: Dict[str, Any]):
    input_path = config['INPUT_VIDEO']
    output_path = config['OUTPUT_VIDEO']
    speed_factor = config['SPEED_FACTOR']

    if speed_factor <= 1.0:
        raise ValueError("加速倍数 SPEED_FACTOR 必须大于 1.0！")

    # 1. 打开输入视频流
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开输入视频: {input_path}")

    # 2. 读取原始视频元数据
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 防止部分视频获取 FPS 失败
    fps = orig_fps if (orig_fps > 0 and orig_fps == orig_fps) else 25.0

    # 3. 创建输出视频写入器（保持原视频的分辨率和帧率）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

    print(f"--> 成功载入视频: {input_path}")
    print(f"    原始分辨率: {orig_w}x{orig_h} | 原始帧率: {fps} FPS | 总帧数: {total_frames}")
    print(f"--> 开始进行 {speed_factor} 倍速抽帧加速处理...")

    start_time = time.time()
    saved_frame_count = 0

    try:
        while True:
            # 根据加速倍数，计算下一个应该抽取的精准帧索引
            target_frame_idx = int(saved_frame_count * speed_factor)

            if target_frame_idx >= total_frames:
                break

            # 利用 CAP_PROP_POS_FRAMES 强行让硬件解码器指针瞬移
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

            ret, frame = cap.read()
            if not ret:
                break

            # 写入视频
            writer.write(frame)
            saved_frame_count += 1

            # 每隔 10% 的进度打印一次日志，避免刷屏
            percent = (target_frame_idx / total_frames) * 100
            if saved_frame_count % max(1, int(total_frames / speed_factor / 10)) == 0:
                print(f"    处理进度: {percent:.1f}% | 已写入: {saved_frame_count} 帧")

        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n==================== 视频处理报告 ====================")
        print(f" ✅ 视频加速完成！")
        print(f" 📂 结果保存至: {output_path}")
        print(f" 🎞️ 生成总帧数: {saved_frame_count} 帧")
        print(f" ⏱️ 任务总耗时: {elapsed_time:.2f} 秒")
        print("====================================================\n")

    except Exception as e:
        print(f"❌ 视频加速流水线发生异常: {str(e)}")
    finally:
        cap.release()
        writer.release()
        print("--> 视频流资源已安全释放。")


if __name__ == '__main__':
    accelerate_video(VIDEO_CONFIG)
