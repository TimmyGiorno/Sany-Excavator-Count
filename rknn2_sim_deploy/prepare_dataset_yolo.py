import cv2
import os

# 1. 路径配置
SOURCE_IMG_DIR = './raw_imgs_yolo'  # 原始场景图片文件夹
OUTPUT_IMG_DIR = './calibration_imgs_yolo'  # 处理后输出的文件夹
DATASET_TXT = './dataset.txt'  # 生成给 RKNN 用的 txt 文件路径
TARGET_SHAPE = (320, 320)  # 模型输入尺寸


def letterbox(input_img, new_shape=(640, 640), color=(114, 114, 114)):
    """与推理代码中完全一致的 letterbox 函数"""
    shape = input_img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        input_img = cv2.resize(input_img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    output_img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return output_img


def main():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    # 支持常见的图片格式
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [os.path.join(SOURCE_IMG_DIR, f) for f in os.listdir(SOURCE_IMG_DIR)
                   if f.lower().endswith(valid_exts)]

    if not image_paths:
        print(f"❌ 错误：在 {SOURCE_IMG_DIR} 中没有找到图片！")
        return

    # 选取前 200 张即可，多了量化速度慢且收益不大
    image_paths = image_paths[:200]

    print(f"--> 开始处理 {len(image_paths)} 张校准图片...")

    with open(DATASET_TXT, 'w') as txt_file:
        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. 应用 letterbox 预处理
            processed_img = letterbox(img, new_shape=TARGET_SHAPE)

            # 2. 保存处理好的图片 (保存为 BGR 格式，OpenCV 默认)
            base_name = os.path.basename(img_path)
            save_path = os.path.join(OUTPUT_IMG_DIR, base_name)
            cv2.imwrite(save_path, processed_img)

            # 3. 将路径写入 .txt 文件
            txt_file.write(f"{save_path.replace(chr(92), '/')}\n")

            if (i + 1) % 10 == 0:
                print(f"    已处理 {i + 1}/{len(image_paths)} 张...")

    print(f"✅ 处理完成！已生成 {DATASET_TXT}，请将此文件路径填入 rknn.build(dataset=...) 中。")


if __name__ == '__main__':
    main()
