import cv2
import time
from rknn.api import RKNN
import numpy as np

# I/O 参数
MODEL_PATH = './best.onnx' # ONNX 模型路径
DATASET_PATH = './dataset.txt' # 用于量化的校准数据集集
INPUT_VIDEO = './test_video.mp4' # 输入的测试视频
OUTPUT_RKNN = './yolov_model_rk3568.rknn'

# 与 train:datasets:data.yaml 中的保持一致
CLASSES = ['bucket-empty', 'bucket-full', 'truck-empty', 'truck-full']

# 输入视频流的实际宽高
MODEL_INPUT_WIDTH = 960
MODEL_INPUT_HEIGHT = 544


def letterbox(input_img, new_shape=(480, 480), color=(114, 114, 114)):
    """
    YOLO 官方的图片缩放方式：等比例缩放，并用灰色填充不足的部分

    Args:
        input_img: 输入图像 (H, W, C)
        new_shape: 目标尺寸 (Height, Width)
        color: 填充色
    """
    shape = input_img.shape[:2]  # 当前图片的 [高, 宽]

    # 计算缩放比例 (取宽和高中较小的缩放比例，以保证能放进目标尺寸)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 计算需要填充的黑边/灰边大小
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # 宽方向两边各填充一半
    dh /= 2  # 高方向两边各填充一半

    # 等比例缩放
    if shape[::-1] != new_unpad:
        input_img = cv2.resize(input_img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 2. 填充边框 (默认填充 YOLO 的背景灰 114)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    output_img = cv2.copyMakeBorder(input_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return output_img, r, (dw, dh)


def post_process(outputs, ratio, pad_w, pad_h, conf_threshold=0.25, iou_threshold=0.45):
    """
    解析 YOLO 的输出，并将其映射回原始视频流的坐标系。
    """
    # 去掉 batch 维度
    predictions = outputs[0][0] 
    
    # 转置特征图，每一行代表一个预测框
    predictions = np.transpose(predictions) 

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:
        # 前 4 个值是 cx, cy, w, h；后面的是多个类的置信度
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        max_score = class_scores[class_id]

        if max_score > conf_threshold:
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]

            # 逆向 LetterBox，减去之前填充的灰边 (pad_w, pad_h)，再除以缩放比例 ratio，还原到原始视频流的像素尺寸
            xmin = ((cx - w / 2) - pad_w) / ratio
            ymin = ((cy - h / 2) - pad_h) / ratio
            box_width = w / ratio
            box_height = h / ratio

            # OpenCV NMS 需要的格式是 [x, y, width, height]
            boxes.append([int(xmin), int(ymin), int(box_width), int(box_height)])
            scores.append(float(max_score))
            class_ids.append(class_id)

    # 利用 OpenCV 自带的 NMS 功能去除重叠框
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            results.append({
                "xmin": x, 
                "ymin": y, 
                "xmax": x + bw, 
                "ymax": y + bh,
                "score": scores[i],
                "class_id": class_ids[i]
            })
    return results


def draw_boxes(frame, results):
    """
    在原始画面上绘制检测框
    """
    for res in results:
        xmin, ymin, xmax, ymax = res['xmin'], res['ymin'], res['xmax'], res['ymax']
        score = res['score']
        class_id = res['class_id']
        
        # 防止越界
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(frame.shape[1], xmax), min(frame.shape[0], ymax)

        # 获取类别名称（如果 class_id 超出列表范围，则显示 ID）
        label = CLASSES[class_id] if class_id < len(CLASSES) else f"Class {class_id}"
        text = f"{label}: {score:.2f}"

        # 画框 (绿色，线宽 2)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # 画标签底色和文字
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (xmin, ymin - text_h - baseline), (xmin + text_w, ymin), (0, 255, 0), -1)
        cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)



if __name__ == '__main__':
    rknn = RKNN(verbose=False)

    # 1. 配置模型参数 (mean_values/std_values 必须与训练时一致)
    rknn.config(mean_values=[[0, 0, 0]], 
                std_values=[[255, 255, 255]], 
                target_platform='rk3568')

    # 2. 加载 ONNX 模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=MODEL_PATH)
    if ret != 0:
        print('Load model failed!')
        exit(ret)

    # 3. 可选：构建模型进行 INT8 量化
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)

    # 4. 初始化模拟器运行时
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime failed!')
        exit(ret)


    print('--> Starting Video Inference')
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: Cannot open video {INPUT_VIDEO}")
        exit(-1)


    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # 防止获取不到帧率
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('./result_visualization.mp4', fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    total_infer_time = 0.0

    # 循环读取视频帧，如果要读完，就改成 while True
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video reached the end or failed to read.")
            break
            
        frame_count += 1

        img, ratio, (dw, dh) = letterbox(frame, new_shape=(480, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 颜色空间转换 BGR -> RGB
        img = np.expand_dims(img, axis=0) # 增加 Batch 维度


        start_time = time.time()
        outputs = rknn.inference(inputs=[img])
        end_time = time.time()
        infer_time = (end_time - start_time) * 1000  # 转换为毫秒
        total_infer_time += infer_time
        print(f"Frame {frame_count:04d} | Inference time: {infer_time:.2f} ms")

        results = post_process(outputs, ratio, dw, dh, conf_threshold=0.3, iou_threshold=0.45)
        
        # 在原画面上绘制结果
        draw_boxes(frame, results)
        
        # 左上角打印一下帧率信息
        cv2.putText(frame, f"Simul Time: {infer_time:.2f} ms", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 写入生成好的视频文件
        out_video.write(frame)

    # 计算平均性能 (仅作参考，PC 模拟器的性能不代表真实 NPU 性能)
    if frame_count > 0:
        print(f"\n[Summary] Processed {frame_count} frames. Average Inference Time: {total_infer_time/frame_count:.2f} ms/frame")

    print('--> Export rknn model')
    rknn.export_rknn(OUTPUT_RKNN)

    cap.release()
    rknn.release()
