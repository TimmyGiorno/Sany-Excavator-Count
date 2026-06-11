import cv2
import time
import numpy as np
from rknn.api import RKNN
from typing import List, Tuple, Dict, Any

# ================= 配置区 =================
MODEL_CONFIG = {
    # YOLO 检测模型参数
    'YOLO_ONNX_PATH': './tmp_files/best.onnx',
    'YOLO_OUTPUT_RKNN': './tmp_files/best.rknn',
    'YOLO_DATASET_PATH': './dataset_yolo.txt',
    'INPUT_SIZE': (640, 640),
    'CLASSES': ['bucket-empty', 'bucket-full', 'truck', 'loading', 'dumping'],
    'REG_MAX': 16,

    # 孪生网络重识别模型参数
    'SIAMESE_ONNX_PATH': './tmp_files/siamese_extractor.onnx',
    'SIAMESE_OUTPUT_RKNN': './tmp_files/siamese_extractor.rknn',
    'SIAMESE_DATASET_PATH': './dataset_siamese.txt',
    'SIAMESE_INPUT_SIZE': (224, 224),
    'SIAMESE_THRESH': 0.75,  # 判定为同一辆车的重识别置信度阈值

    # 流水线与业务核心参数
    'INPUT_VIDEO': './tmp_files/shift_cut.mp4',
    'OUTPUT_VIDEO': './tmp_files/result_visualization.mp4',
    'NMS_OFFSET': 4096,
    'CONF_THRESH': 0.3,
    'IOU_THRESH': 0.45,
    'MAX_FRAMES': 0,  # 设为 > 0 用于截断快速测试，设为 0 则跑完整个视频
}

CLASS_COLORS = {
    'bucket-empty': (255, 0, 0),
    'bucket-full': (0, 0, 255),
    'truck': (0, 255, 255),
    'loading': (0, 255, 0),
    'dumping': (255, 0, 255),
}


# ================= 核心算法层 =================
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


def check_horizontal_overlap(box1: List[int], box2: List[int]) -> bool:
    """检查两个边界框在水平方向上是否有交集"""
    return not (box1[2] < box2[0] or box2[2] < box1[0])


def expand_bbox_for_truck(box: List[int], frame_shape: Tuple[int, int, int]) -> List[int]:
    """将不规则的卡车边界框智能扩展对齐为正方形，以便契合重识别网络的输入特征轴"""
    h, w = frame_shape[:2]
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    if width > height:
        new_height = width
        ymin = max(0, ymax - new_height)
    else:
        new_width = height
        total_dx = new_width - width
        left_dx = total_dx // 2
        right_dx = total_dx - left_dx
        xmin = max(0, xmin - left_dx)
        xmax = min(w, xmax + right_dx)
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


# ================= 主控制流程 =================
def run_inference():
    """主推理流程控制函数"""

    rknn_yolo = RKNN(verbose=False)
    rknn_siamese = RKNN(verbose=False)

    state = {
        'ticket_id': None,
        'total_truck_count': 0,
        'total_bucket_count': 0,
        'base_bucket_count': 0,
        'bucket_full': False,
        'dumping_active': False,
        'last_dumping_box': None,
        'reference_truck_img': None,
        'reference_truck_emb': None,
    }

    try:
        # 1. 初始化 YOLO 引擎
        print("--> 正在编译量化组件（YOLO 目标检测引擎）...")
        rknn_yolo.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3568',
                         quant_img_RGB2BGR=True)
        if rknn_yolo.load_onnx(model=MODEL_CONFIG['YOLO_ONNX_PATH']) != 0:
            raise RuntimeError("加载 YOLO ONNX 失败")
        ret = rknn_yolo.build(do_quantization=True, dataset=MODEL_CONFIG['YOLO_DATASET_PATH'])
        if ret != 0:
            raise RuntimeError("YOLO 模型量化失败")
        rknn_yolo.export_rknn(MODEL_CONFIG['YOLO_OUTPUT_RKNN']) # 导出 yolo 的 RKNN 模型
        if rknn_yolo.init_runtime() != 0:
            raise RuntimeError("启动 YOLO 硬件环境失败")

        # 2. 初始化 Siamese 引擎
        print("--> 正在编译量化组件（Siamese 车辆重识别引擎）...")
        rknn_siamese.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.375]],
                            target_platform='rk3568', quant_img_RGB2BGR=True)
        if rknn_siamese.load_onnx(model=MODEL_CONFIG['SIAMESE_ONNX_PATH'], inputs=['input_image'],
                                  input_size_list=[[1, 3, 224, 224]]) != 0:
            raise RuntimeError("加载 Siamese ONNX 失败")
        ret = rknn_siamese.build(do_quantization=True, dataset=MODEL_CONFIG['SIAMESE_DATASET_PATH'])
        if ret != 0:
            raise RuntimeError("Siamese 模型量化失败")
        rknn_siamese.export_rknn(MODEL_CONFIG['SIAMESE_OUTPUT_RKNN']) # 导出 siamese 的 RKNN 模型
        if rknn_siamese.init_runtime() != 0:
            raise RuntimeError("启动 Siamese 硬件环境失败")

        # 3. 视频流输入输出初始化
        cap = cv2.VideoCapture(MODEL_CONFIG['INPUT_VIDEO'])
        if not cap.isOpened():
            raise ValueError("视频流载入阻断：文件无法打开")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (0 < fps == fps) else 25.0

        display_width = orig_h // 2
        canvas_w = orig_w + display_width + 20
        writer = cv2.VideoWriter(MODEL_CONFIG['OUTPUT_VIDEO'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, orig_h))

        frame_count, total_infer_time = 0, 0.0
        max_frames_limit = MODEL_CONFIG['MAX_FRAMES']

        print("--> 开始音视频帧逐帧推理...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if 0 < max_frames_limit < frame_count:
                print(f"--> 已达到设定的最大截断帧数 limit ({max_frames_limit} 帧)，提前终止处理。")
                frame_count -= 1
                break

            img, ratio, (dw, dh) = letterbox(frame, new_shape=MODEL_CONFIG['INPUT_SIZE'])
            img_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[None, ...]

            start_time = time.time()
            outputs = rknn_yolo.inference(inputs=[img_tensor])
            infer_time_ms = (time.time() - start_time) * 1000
            total_infer_time += infer_time_ms

            results = post_process(outputs, ratio, dw, dh)

            truck_boxes = [r for r in results if r['class_id'] == 2]
            bucket_boxes = [r for r in results if r['class_id'] in [0, 1]]
            dumping_boxes = [r for r in results if r['class_id'] == 4]

            # === 第二阶段：业务逻辑状态机 ===

            # A. 铲斗绝对全局累加
            if bucket_boxes:
                best_bucket = max(bucket_boxes, key=lambda x: x['score'])
                b_box = [best_bucket['xmin'], best_bucket['ymin'], best_bucket['xmax'], best_bucket['ymax']]

                if best_bucket['class_id'] == 1:
                    if not state['bucket_full']:
                        has_overlap = any(
                            check_horizontal_overlap(b_box, [t['xmin'], t['ymin'], t['xmax'], t['ymax']]) for t in
                            truck_boxes)
                        if not has_overlap:
                            state['bucket_full'] = True

                elif best_bucket['class_id'] == 0:
                    if state['bucket_full']:
                        has_overlap = any(
                            check_horizontal_overlap(b_box, [t['xmin'], t['ymin'], t['xmax'], t['ymax']]) for t in
                            truck_boxes)
                        if has_overlap:
                            state['bucket_full'] = False
                            state['total_bucket_count'] += 1
                        else:
                            state['bucket_full'] = False

            # B. 卡车卸料与票号系统
            has_dumping = len(dumping_boxes) > 0

            if has_dumping and not state['bucket_full']:
                pass
            else:
                if has_dumping and not state['dumping_active']:
                    state['dumping_active'] = True

                if has_dumping:
                    state['last_dumping_box'] = [dumping_boxes[0]['xmin'], dumping_boxes[0]['ymin'],
                                                 dumping_boxes[0]['xmax'], dumping_boxes[0]['ymax']]

                # 倒土彻底结束且土已入车，触发截图比对
                if not has_dumping and state['dumping_active'] and not state['bucket_full']:
                    if state['last_dumping_box'] is not None and truck_boxes:
                        ld_box = state['last_dumping_box']
                        dump_cx = (ld_box[0] + ld_box[2]) / 2
                        closest_truck, min_dist = None, float('inf')

                        for truck in truck_boxes:
                            t_box = [truck['xmin'], truck['ymin'], truck['xmax'], truck['ymax']]
                            if check_horizontal_overlap(ld_box, t_box):
                                dist = abs(((t_box[0] + t_box[2]) / 2) - dump_cx)
                                if dist < min_dist:
                                    min_dist, closest_truck = dist, truck

                        if closest_truck:
                            t_box = [closest_truck['xmin'], closest_truck['ymin'], closest_truck['xmax'],
                                     closest_truck['ymax']]
                            exp_box = expand_bbox_for_truck(t_box, frame.shape)
                            truck_crop = frame[exp_box[1]:exp_box[3], exp_box[0]:exp_box[2]]

                            if truck_crop.size > 0:
                                siamese_img = cv2.resize(truck_crop, MODEL_CONFIG['SIAMESE_INPUT_SIZE'],
                                                         interpolation=cv2.INTER_LINEAR)
                                siamese_tensor = np.expand_dims(siamese_img, axis=0)

                                current_emb = rknn_siamese.inference(inputs=[siamese_tensor])[0][0]

                                # 票号与重识别判定
                                if state['reference_truck_emb'] is not None:
                                    similarity = float(np.dot(state['reference_truck_emb'], current_emb))
                                    if similarity < MODEL_CONFIG['SIAMESE_THRESH']:
                                        # 换新车
                                        state['total_truck_count'] += 1
                                        state['ticket_id'] = f"TKT_{int(time.time() * 1000)}"
                                        # 换车发生在第一铲之后，所以基数为当前总数 - 1
                                        state['base_bucket_count'] = max(0, state['total_bucket_count'] - 1)
                                        print(
                                            f"-> 帧号: {frame_count:04d} | [新车入场] 相似度: {similarity:.2f}, 票号: {state['ticket_id']}")
                                    else:
                                        print(f"-> 帧号: {frame_count:04d} | [原目标存留] 相似度: {similarity:.2f}")
                                else:
                                    # 系统启动第一辆车
                                    state['total_truck_count'] += 1
                                    state['ticket_id'] = f"TKT_{int(time.time() * 1000)}"
                                    state['base_bucket_count'] = max(0, state['total_bucket_count'] - 1)
                                    print(
                                        f"-> 帧号: {frame_count:04d} | [系统启动] 捕获首辆卡车, 票号: {state['ticket_id']}")

                                # 无条件滚动更新特征，防止特征漂移
                                state['reference_truck_img'] = truck_crop.copy()
                                state['reference_truck_emb'] = current_emb

                    state['dumping_active'] = False

            # === 第三阶段：最终可视化渲染 ===

            # 计算业务层单车斗数
            current_truck_buckets = 0
            if state['ticket_id'] is not None:
                current_truck_buckets = max(0, state['total_bucket_count'] - state['base_bucket_count'])

            for res in results:
                label = MODEL_CONFIG['CLASSES'][res['class_id']]
                color = CLASS_COLORS.get(label, (255, 255, 255))
                cv2.rectangle(frame, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), color, 2)
                text = f"{label}: {res['score']:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (res['xmin'], res['ymin'] - th - 5), (res['xmin'] + tw, res['ymin']), color, -1)
                cv2.putText(frame, text, (res['xmin'], res['ymin'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 扩展了左上角面板以容纳票号
            cv2.rectangle(frame, (20, 20), (450, 150), (0, 0, 0), -1)
            cv2.putText(frame, f"Trucks: {state['total_truck_count']}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"Cur Buckets: {current_truck_buckets}", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            ticket_text = state['ticket_id'] if state['ticket_id'] else "Waiting..."
            cv2.putText(frame, f"ID: {ticket_text}", (35, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            output_canvas = np.zeros((orig_h, canvas_w, 3), dtype=np.uint8)
            output_canvas[0:orig_h, 0:orig_w] = frame

            right_x = orig_w + 10
            block_h = orig_h // 2 - 15

            # 右侧 UI 简化为只显示锁定的特征图
            if state['reference_truck_img'] is not None:
                p_resize = cv2.resize(state['reference_truck_img'], (display_width, block_h))
                output_canvas[10:10 + block_h, right_x:right_x + display_width] = p_resize
                cv2.putText(output_canvas, "Locked Target", (right_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            else:
                cv2.rectangle(output_canvas, (right_x, 10), (right_x + display_width, 10 + block_h), (40, 40, 40), -1)
                cv2.putText(output_canvas, "No Target", (right_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (150, 150, 150), 1)

            writer.write(output_canvas)

            if frame_count > 0 and frame_count % 10 == 0:
                avg_time = total_infer_time / frame_count
                print(
                    f"帧数: {frame_count:04d} | 单帧组合推理耗时: {infer_time_ms:.2f} ms | 当前平均: {avg_time:.2f} ms")

        if frame_count > 0:
            avg_time = total_infer_time / frame_count
            print("\n==================== 性能分析报告 ====================")
            print(f" 📊 成功处理视频总帧数: {frame_count} 帧")
            print(f" ⏱️ 平均单帧仿真耗时: {avg_time:.2f} ms")
            print(f" 🚀 主流程综合计算速度: {1000.0 / avg_time:.2f} FPS")
            print("===============================================================\n")

    except Exception as e:
        import traceback
        print(f"❌ 流水线运行时异常捕获:\n{traceback.format_exc()}")
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'writer' in locals(): writer.release()
        rknn_yolo.release()
        rknn_siamese.release()
        print("--> 双向计算硬件资源安全回收。")


if __name__ == '__main__':
    run_inference()
