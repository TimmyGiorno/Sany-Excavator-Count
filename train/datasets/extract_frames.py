import cv2
import os


def extract_frames(video_path: str, output_dir: str, interval: int = 10):
    """
    将视频按帧拆分为图片，用于训练 YOLO 模型。

    :param video_path: 原始视频文件的相对或绝对路径
    :param output_dir: 拆分后图片保存的目录
    :param interval: 抽帧间隔
    """

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[train:datasets:extract_frames] 错误: 无法打开视频文件 '{video_path}'，请检查路径是否正确。")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[train:datasets:extract_frames] 成功打开视频。总帧数: {total_frames}")
    print(f"[train:datasets:extract_frames] 准备将图片导出至: {output_dir}")

    frame_index = 0  # 视频当前的真实帧数
    saved_count = 0  # 实际保存下来的图片数量

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % interval == 0:
            filename = f"frame_{frame_index:06d}.png"
            output_path = os.path.join(output_dir, filename)

            success = cv2.imwrite(output_path, frame)
            # print(f"保存 {'成功' if success else '失败'}: {output_path}")
            saved_count += 1

            if saved_count % 50 == 0:
                print(
                    f"[train:datasets:extract_frames] 已保存 {saved_count} 张图片 (当前正处理视频第 {frame_index}/{total_frames} 帧)...")

        frame_index += 1

    cap.release()
    print(
        f"[train:datasets:extract_frames] 抽取完成！视频共 {total_frames} 帧，按每 {interval} 帧抽 1 张，最终得到 {saved_count} 张图片。")


# if __name__ == "__main__":
#     VIDEO_FILE = "E:/data/datasetsV1.1/JFSK_20251230_115914_N1_00/JFSK_20251230_115914_N1_00.mp4"
#     OUTPUT_FOLDER = "E:/data/datasetsV1.1/JFSK_20251230_115914_N1_00/images/train"
#
#     extract_frames(VIDEO_FILE, OUTPUT_FOLDER)
if __name__ == "__main__":
    # 定义基础路径和视频名称
    BASE_PATH = "E:/pycharmProjects/Sany-Excavator-Count/train/datasets/datasetV1.3/"
    VIDEO_NAMES = [
        "JFSK_20251230_110914_N1_00",
        "JFSK_20251230_113914_N1_00",
        "JFSK_20251230_115914_N1_00",
        "JFSK_20251230_120914_N1_00",
        "JFSK_20251230_121914_N1_00",
        "JFSK_20251230_154914_N1_00",
        "JFSK_20251230_155914_N1_00",
        "JFSK_20251230_165914_N1_00",
        "JFSK_20251230_170914_N1_00",
        "JFSK_20251230_172915_N1_00",
        "JFSK_20251231_004916_N1_00"
    ]

    # 自动生成路径
    for video_name in VIDEO_NAMES:
        video_file = f"{BASE_PATH}{video_name}/{video_name}.mp4"
        output_folder = f"{BASE_PATH}{video_name}/images/train"

        print(f"\n处理视频: {video_file}")
        print(f"输出到: {output_folder}")
        extract_frames(video_file, output_folder)