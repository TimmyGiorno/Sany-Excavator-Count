import cv2
import os


def extract_frames(video_path, output_dir, interval=10):
    """
    将视频按帧拆分为图片
    :param video_path: 原始视频文件的相对或绝对路径
    :param output_dir: 拆分后图片保存的目录
    :param interval: 抽帧间隔
    """
    # 1. 如果指定的输出文件夹不存在，则自动创建它（包括多层目录）
    os.makedirs(output_dir, exist_ok=True)

    # 2. 读取视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频文件 '{video_path}'，请检查路径是否正确。")
        return

    # 获取视频的总帧数（仅供进度参考）
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f">>> 成功打开视频。总帧数: {total_frames}")
    print(f">>> 准备将图片导出至: {output_dir}")

    frame_index = 0  # 视频当前的真实帧数
    saved_count = 0  # 实际保存下来的图片数量

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 核心逻辑：只有当当前帧数能被 interval 整除时，才保存图片
        if frame_index % interval == 0:
            # 这里的命名依然使用视频的真实帧索引，保证能和标签对得上
            filename = f"frame_{frame_index:06d}.png"
            output_path = os.path.join(output_dir, filename)

            cv2.imwrite(output_path, frame)
            saved_count += 1

            # 打印进度提示
            if saved_count % 50 == 0:
                print(f"已保存 {saved_count} 张图片 (当前正处理视频第 {frame_index}/{total_frames} 帧)...")

        frame_index += 1

    cap.release()
    print(f"\n✅ 抽取完成！视频共 {total_frames} 帧，按每 {interval} 帧抽 1 张，最终得到 {saved_count} 张图片。")


if __name__ == "__main__":
    # ==========================================
    # 请在这里修改你的实际路径配置
    # ==========================================

    # 你的原始视频在哪里？
    VIDEO_FILE = "night/JFSK_20251230_165914_N1_00.mp4"  # <--- 替换为你自己的视频文件名/路径

    # 你想把图片存到哪里？(脚本会自动帮你建好 data.yaml/images/train 这个目录)
    OUTPUT_FOLDER = "images/train"

    extract_frames(VIDEO_FILE, OUTPUT_FOLDER)