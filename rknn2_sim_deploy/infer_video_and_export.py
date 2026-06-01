import cv2
import time
import numpy as np
from rknn.api import RKNN
from typing import List, Tuple, Dict, Any

# ================= 配置区 =================
MODEL_CONFIG = {
    'ONNX_PATH': './tmp_files/best.onnx',
    'DATASET_PATH': './dataset.txt',
    'INPUT_VIDEO': './tmp_files/test_video.mp4',
    'OUTPUT_RKNN': './tmp_files/best.rknn',
    'OUTPUT_VIDEO': './tmp_files/result_visualization.mp4',
    'CLASSES': ['bucket-empty', 'bucket-full', 'truck', 'loading', 'dumping'],
    'INPUT_SIZE': (640, 640),
    'NMS_OFFSET': 4096,
    'CONF_THRESH': 0.3,
    'IOU_THRESH': 0.45,
    'REG_MAX': 16,
    'MAX_FRAMES': 1000,  # 帧数限制：设为 > 0 的数字表示最多处理的帧数（方便测试），设为 0 表示不限制处理完整视频
}

CLASS_COLORS = {
    'bucket-empty': (255, 0, 0),
    'bucket-full': (0, 0, 255),
    'truck': (0, 255, 255),
    'loading': (0, 255, 0),
    'dumping': (255, 0, 255),
}


# ==========================================

def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """对图像进行等比例缩放并填充，适配模型输入"""
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    output_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return output_img, ratio, (dw, dh)


def dfl_decode(position: np.ndarray) -> np.ndarray:
    """对 DFL 分布进行解码，计算预测框的边界偏移量"""
    x = position - np.max(position, axis=-1, keepdims=True)
    softmax_x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    dfl_weights = np.arange(MODEL_CONFIG['REG_MAX'], dtype=np.float32)
    return np.sum(softmax_x * dfl_weights, axis=-1)


def post_process(outputs: List[np.ndarray], ratio: float, pad_w: float, pad_h: float) -> List[Dict[str, Any]]:
    """YOLOv8/11 后处理：包含 DFL 解码、逆缩放及 Class-Aware NMS"""
    boxes, scores, class_ids = [], [], []
    strides = [8, 16, 32]
    reg_max = MODEL_CONFIG['REG_MAX']
    num_classes = len(MODEL_CONFIG['CLASSES'])

    for i, stride in enumerate(strides):
        output = outputs[i][0]

        if output.shape[0] == 4 * reg_max + num_classes:
            output = output.transpose(1, 2, 0)

        cls_scores = output[..., 4 * reg_max:]
        reg_preds = output[..., :4 * reg_max]

        cls_scores = np.clip(cls_scores, -88.0, 88.0)
        cls_scores = 1 / (1 + np.exp(-cls_scores))

        y, x, c = np.where(cls_scores > MODEL_CONFIG['CONF_THRESH'])
        if len(y) == 0:
            continue

        valid_reg_preds = reg_preds[y, x]
        valid_reg_preds = valid_reg_preds.reshape(-1, 4, reg_max)

        pred_ltrb = dfl_decode(valid_reg_preds)

        x1 = ((x + 0.5 - pred_ltrb[:, 0]) * stride - pad_w) / ratio
        y1 = ((y + 0.5 - pred_ltrb[:, 1]) * stride - pad_h) / ratio
        x2 = ((x + 0.5 + pred_ltrb[:, 2]) * stride - pad_w) / ratio
        y2 = ((y + 0.5 + pred_ltrb[:, 3]) * stride - pad_h) / ratio

        for j in range(len(x1)):
            boxes.append([int(x1[j]), int(y1[j]), int(x2[j] - x1[j]), int(y2[j] - y1[j])])
            scores.append(float(cls_scores[y[j], x[j], c[j]]))
            class_ids.append(int(c[j]))

    if not boxes:
        return []

    boxes_nms = [[b[0] + class_ids[i] * MODEL_CONFIG['NMS_OFFSET'],
                  b[1] + class_ids[i] * MODEL_CONFIG['NMS_OFFSET'],
                  b[2], b[3]] for i, b in enumerate(boxes)]

    indices = cv2.dnn.NMSBoxes(boxes_nms, scores, MODEL_CONFIG['CONF_THRESH'], MODEL_CONFIG['IOU_THRESH'])

    if len(indices) == 0:
        return []

    return [{"xmin": boxes[i][0], "ymin": boxes[i][1], "xmax": boxes[i][0] + boxes[i][2],
             "ymax": boxes[i][1] + boxes[i][3], "score": scores[i], "class_id": class_ids[i]}
            for i in indices.flatten()]


def run_inference():
    """主推理流程控制函数"""
    rknn = RKNN(verbose=False)

    try:
        # 1. 模型配置与量化编译
        print("--> 正在初始化 RKNN 配置...")
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3568',
                    quant_img_RGB2BGR=True)

        print(f"--> 正在加载 ONNX 模型: {MODEL_CONFIG['ONNX_PATH']}")
        ret = rknn.load_onnx(model=MODEL_CONFIG['ONNX_PATH'])
        if ret != 0: raise RuntimeError("加载 ONNX 模型失败！")

        print("--> 正在构建模型并进行 INT8 量化校准（这需要一些时间）...")
        ret = rknn.build(do_quantization=True, dataset=MODEL_CONFIG['DATASET_PATH'])
        if ret != 0: raise RuntimeError("构建 RKNN 模型失败！")

        print(f"--> 正在导出 RKNN 模型至: {MODEL_CONFIG['OUTPUT_RKNN']}")
        ret = rknn.export_rknn(MODEL_CONFIG['OUTPUT_RKNN'])
        if ret != 0: raise RuntimeError("导出 RKNN 模型文件失败！")

        print("--> 正在初始化 Runtime 仿真运行环境...")
        ret = rknn.init_runtime()
        if ret != 0: raise RuntimeError("初始化 Runtime 失败！")

        # 2. 视频流输入输出初始化
        print("--> 正在打开输入视频流...")
        cap = cv2.VideoCapture(MODEL_CONFIG['INPUT_VIDEO'])
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {MODEL_CONFIG['INPUT_VIDEO']}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps > 0 and fps == fps) else 25.0

        writer = cv2.VideoWriter(MODEL_CONFIG['OUTPUT_VIDEO'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

        frame_count = 0
        total_infer_time = 0.0
        max_frames_limit = MODEL_CONFIG['MAX_FRAMES']

        print("--> 开始音视频帧逐帧推理...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("--> 视频流读取完毕或读取结束。")
                break

            frame_count += 1

            if max_frames_limit > 0 and frame_count > max_frames_limit:
                print(f"--> 已达到设定的最大截断帧数 limit ({max_frames_limit} 帧)，提前终止处理。")
                frame_count -= 1  # 修正计数用于后续平均计算
                break

            img, ratio, (dw, dh) = letterbox(frame, new_shape=MODEL_CONFIG['INPUT_SIZE'])
            img_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None, ...]

            # 单帧耗时统计开始
            start_time = time.time()
            outputs = rknn.inference(inputs=[img_tensor])
            end_time = time.time()

            infer_time_ms = (end_time - start_time) * 1000
            total_infer_time += infer_time_ms
            if frame_count % 10 == 0:
                print(f"帧数: {frame_count:04d} | 单帧 NPU 仿真推理耗时: {infer_time_ms:.2f} ms")

            results = post_process(outputs, ratio, dw, dh)

            for res in results:
                label = MODEL_CONFIG['CLASSES'][res['class_id']]
                color = CLASS_COLORS.get(label, (255, 255, 255))
                cv2.rectangle(frame, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), color, 2)

                text = f"{label}: {res['score']:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (res['xmin'], res['ymin'] - th - 5), (res['xmin'] + tw, res['ymin']), color, -1)
                cv2.putText(frame, text, (res['xmin'], res['ymin'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            writer.write(frame)

        # 3. 统计结果最终汇总打印报告
        if frame_count > 0:
            avg_time = total_infer_time / frame_count
            print("\n==================== 性能分析报告 (仿真环境) ====================")
            print(f" 📊 成功处理视频总帧数: {frame_count} 帧")
            print(f" ⏱️ 平均单帧仿真推理耗时: {avg_time:.2f} ms")
            print(f" 🚀 理论仿真吞吐量速率: {1000.0 / avg_time:.2f} FPS")
            print("===============================================================\n")
            print(f"--> 视频可视化结果已保存在: {MODEL_CONFIG['OUTPUT_VIDEO']}")

    except Exception as e:
        print(f"❌ 商业流水线运行时异常捕获: {str(e)}")

    finally:
        # 安全释放资源防止后台进程死锁或 C 层内存泄漏
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'writer' in locals(): writer.release()
        rknn.release()
        print("--> 硬件运行时资源已安全回收释放。")


if __name__ == '__main__':
    run_inference()