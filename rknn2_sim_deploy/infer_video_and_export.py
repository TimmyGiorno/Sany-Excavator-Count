import cv2
import time
import numpy as np
from rknn.api import RKNN
from typing import List, Tuple, Dict, Any

# ================= 配置区 =================
MODEL_CONFIG = {
    'YOLO_ONNX_PATH': './tmp_files/best.onnx',
    'YOLO_OUTPUT_RKNN': './tmp_files/best.rknn',
    'YOLO_DATASET_PATH': './dataset_yolo.txt',
    'INPUT_SIZE': (640, 640),
    'CLASSES': ['bucket-empty', 'bucket-full', 'truck', 'loading', 'dumping', 'mine'],
    'REG_MAX': 16,

    'INPUT_VIDEO': './tmp_files/test_video_shift_fast.mp4',
    'OUTPUT_VIDEO': './tmp_files/result_visualization.mp4',
    'NMS_OFFSET': 4096,
    'CONF_THRESH': 0.3,
    'IOU_THRESH': 0.45,
    'MAX_FRAMES': 0,
    'TIMEOUT_SEC': 10.0,
    'DECLINE_THRESH': 0.75,
}

CLASS_COLORS = {
    'bucket-empty': (255, 0, 0),
    'bucket-full': (0, 0, 255),
    'truck': (0, 255, 255),
    'loading': (0, 255, 0),
    'dumping': (255, 0, 255),
    'mine': (0, 165, 255),
}


# ================= 核心算法层 =================
def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
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
    x = position - np.max(position, axis=-1, keepdims=True)
    softmax_x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    dfl_weights = np.arange(MODEL_CONFIG['REG_MAX'], dtype=np.float32)
    return np.sum(softmax_x * dfl_weights, axis=-1)


def post_process(outputs: List[np.ndarray], ratio: float, pad_w: float, pad_h: float) -> List[Dict[str, Any]]:
    boxes, scores, class_ids = [], [], []
    strides = [8, 16, 32]
    reg_max = MODEL_CONFIG['REG_MAX']

    for i, stride in enumerate(strides):
        output = outputs[i][0]

        if len(output.shape) == 3 and 64 <= output.shape[0] < 200:
            output = output.transpose(1, 2, 0)

        actual_num_classes = output.shape[-1] - 4 * reg_max

        cls_scores = output[..., 4 * reg_max:]
        reg_preds = output[..., :4 * reg_max]

        cls_scores = np.clip(cls_scores, -88.0, 88.0)
        cls_scores = 1 / (1 + np.exp(-cls_scores))

        y, x, c = np.where(cls_scores > MODEL_CONFIG['CONF_THRESH'])
        if len(y) == 0: continue

        valid_reg_preds = reg_preds[y, x].reshape(-1, 4, reg_max)
        pred_ltrb = dfl_decode(valid_reg_preds)

        x1 = ((x + 0.5 - pred_ltrb[:, 0]) * stride - pad_w) / ratio
        y1 = ((y + 0.5 - pred_ltrb[:, 1]) * stride - pad_h) / ratio
        x2 = ((x + 0.5 + pred_ltrb[:, 2]) * stride - pad_w) / ratio
        y2 = ((y + 0.5 + pred_ltrb[:, 3]) * stride - pad_h) / ratio

        for j in range(len(x1)):
            boxes.append([int(x1[j]), int(y1[j]), int(x2[j] - x1[j]), int(y2[j] - y1[j])])
            scores.append(float(cls_scores[y[j], x[j], c[j]]))
            class_ids.append(int(c[j]))

    if not boxes: return []

    boxes_nms = [[b[0] + class_ids[i] * MODEL_CONFIG['NMS_OFFSET'],
                  b[1] + class_ids[i] * MODEL_CONFIG['NMS_OFFSET'],
                  b[2], b[3]] for i, b in enumerate(boxes)]

    indices = cv2.dnn.NMSBoxes(boxes_nms, scores, MODEL_CONFIG['CONF_THRESH'], MODEL_CONFIG['IOU_THRESH'])
    if len(indices) == 0: return []

    return [{"xmin": boxes[i][0], "ymin": boxes[i][1], "xmax": boxes[i][0] + boxes[i][2],
             "ymax": boxes[i][1] + boxes[i][3], "score": scores[i], "class_id": class_ids[i]}
            for i in indices.flatten()]


def check_horizontal_overlap(box1: List[int], box2: List[int]) -> bool:
    return not (box1[2] < box2[0] or box2[2] < box1[0])


# ================= 主控制流程 =================
def run_inference():
    rknn_yolo = RKNN(verbose=False)

    state = {
        'ticket_id': 'WAITING',
        'total_truck_count': 0,
        'total_bucket_count': 0,
        'current_truck_buckets': 0,

        'pending_buckets': 0,
        'frames_since_bucket_empty': 0,
        'has_pushed_timeout': False,

        'bucket_full': False,
        'dumping_active': False,
        'dumping_frame_count': 0,
        'last_dumping_box': None,

        'stable_frames_remaining': 0,
        'is_statting': False,
        'stat_frames_remaining': 0,
        'ratio_buffer': [],
        'last_avg_ratio': None,
        'current_truck_xyxy': None,
        'last_dumping_bucket_xyxy': None,

        'is_truck_active': False,
        'last_action_time': time.time(),
        'event_logs': []
    }

    def force_complete_truck():
        if state['is_truck_active']:
            event_msg = f"[END] Truck {state['ticket_id']} completed. Total confirmed: {state['current_truck_buckets']}."
            state['event_logs'].insert(0, event_msg)
            print(f">>> {event_msg}")
            state['is_truck_active'] = False
            state['ticket_id'] = 'WAITING'
            state['current_truck_buckets'] = 0
            state['has_pushed_timeout'] = False
            state['last_avg_ratio'] = None

    try:
        print("--> 正在编译量化组件（YOLO 目标检测引擎）...")
        rknn_yolo.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3568',
                         quant_img_RGB2BGR=True)
        if rknn_yolo.load_onnx(model=MODEL_CONFIG['YOLO_ONNX_PATH']) != 0: raise RuntimeError("加载 YOLO 失败")
        if rknn_yolo.build(do_quantization=True, dataset=MODEL_CONFIG['YOLO_DATASET_PATH']) != 0: raise RuntimeError(
            "YOLO 量化失败")
        rknn_yolo.export_rknn(MODEL_CONFIG['YOLO_OUTPUT_RKNN'])
        if rknn_yolo.init_runtime() != 0: raise RuntimeError("启动 YOLO 硬件环境失败")

        cap = cv2.VideoCapture(MODEL_CONFIG['INPUT_VIDEO'])
        if not cap.isOpened(): raise ValueError("视频流载入阻断")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (0 < fps == fps) else 25.0

        log_panel_width = 400
        canvas_w = orig_w + log_panel_width
        writer = cv2.VideoWriter(MODEL_CONFIG['OUTPUT_VIDEO'], cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, orig_h))

        frame_count, total_infer_time = 0, 0.0
        max_frames_limit = MODEL_CONFIG['MAX_FRAMES']

        print("--> 开始音视频帧逐帧推理...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            if 0 < max_frames_limit < frame_count: break
            now = time.time()
            state['frames_since_bucket_empty'] += 1

            if (state['current_truck_buckets'] > 0 or state['pending_buckets'] > 0) and (
                    now - state['last_action_time'] > MODEL_CONFIG['TIMEOUT_SEC']):
                if not state['has_pushed_timeout']:
                    hb_msg = f"[DB PUSH] Heartbeat for TKT: {state['ticket_id']}, Loads: {state['current_truck_buckets']}, Pend: {state['pending_buckets']}"
                    state['event_logs'].insert(0, hb_msg)
                    print(f">>> {hb_msg}")
                    state['has_pushed_timeout'] = True

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
            mine_boxes = [r for r in results if r['class_id'] == 5]

            # === B. 铲斗满空判断与计数 (严格修复版，恢复原版防抖锁) ===
            if bucket_boxes:
                best_bucket = max(bucket_boxes, key=lambda x: x['score'])
                bb_xyxy = [best_bucket['xmin'], best_bucket['ymin'], best_bucket['xmax'], best_bucket['ymax']]

                if best_bucket['class_id'] == 1:
                    # 1. 发现满铲，触发加速释放并上锁
                    if state['pending_buckets'] > 0:
                        state['current_truck_buckets'] += state['pending_buckets']
                        fast_msg = f"[FAST-RELEASE] Confirmed pending {state['pending_buckets']} load(s)."
                        state['event_logs'].insert(0, fast_msg)
                        print(f">>> {fast_msg}")
                        state['pending_buckets'] = 0

                    if not state['bucket_full']:
                        has_overlap = any(
                            check_horizontal_overlap(bb_xyxy, [t['xmin'], t['ymin'], t['xmax'], t['ymax']]) for t in
                            truck_boxes)
                        if not has_overlap:
                            state['bucket_full'] = True

                elif best_bucket['class_id'] == 0:
                    # 2. 发现空铲，必须消耗掉 bucket_full 锁才能触发计数！
                    if state['bucket_full']:
                        has_overlap = any(
                            check_horizontal_overlap(bb_xyxy, [t['xmin'], t['ymin'], t['xmax'], t['ymax']]) for t in
                            truck_boxes)

                        if has_overlap:
                            # 触发有效的一铲，立刻锁死，防止后续帧连续双倍计数！
                            state['bucket_full'] = False
                            state['total_bucket_count'] += 1

                            state['pending_buckets'] += 1
                            state['frames_since_bucket_empty'] = 0
                            state['last_action_time'] = now
                            state['has_pushed_timeout'] = False

                            if not state['is_truck_active']:
                                state['is_truck_active'] = True
                                state['total_truck_count'] += 1
                                state['ticket_id'] = f"TKT_{int(now * 1000)}"
                                start_msg = f"[START] New Truck: {state['ticket_id']}"
                                state['event_logs'].insert(0, start_msg)
                                print(f">>> {start_msg}")

                            pend_msg = f"[PENDING] Dump overlap detected. Hidden pendings: {state['pending_buckets']}"
                            state['event_logs'].insert(0, pend_msg)
                            print(f">>> {pend_msg}")
                        else:
                            state['bucket_full'] = False

            # === C. 跟踪 Dumping 状态 (仅用于打断/激活面积统计) ===
            has_dumping = len(dumping_boxes) > 0
            if has_dumping:
                state['dumping_frame_count'] += 1
                if state['dumping_frame_count'] >= 3:
                    state['dumping_active'] = True

                if state['stable_frames_remaining'] > 0 or state['is_statting']:
                    state['stable_frames_remaining'] = 0
                    state['is_statting'] = False
                    state['stat_frames_remaining'] = 0
                    state['ratio_buffer'] = []
                    state['current_truck_xyxy'] = None
                    state['last_dumping_bucket_xyxy'] = None
            else:
                state['dumping_frame_count'] = 0
                if state['dumping_active']:
                    state['dumping_active'] = False
                    state['stable_frames_remaining'] = 1
                    if bucket_boxes:
                        bb = max(bucket_boxes, key=lambda x: x['score'])
                        state['last_dumping_bucket_xyxy'] = [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]
                    else:
                        state['last_dumping_bucket_xyxy'] = None

            # === D. 处理稳定期，抓取要统计面积的卡车 ===
            if state['stable_frames_remaining'] > 0:
                state['stable_frames_remaining'] -= 1
                if state['stable_frames_remaining'] == 0:
                    if state['last_dumping_bucket_xyxy'] is None and bucket_boxes:
                        bb = max(bucket_boxes, key=lambda x: x['score'])
                        state['last_dumping_bucket_xyxy'] = [bb['xmin'], bb['ymin'], bb['xmax'], bb['ymax']]

                    if state['last_dumping_bucket_xyxy'] is not None and truck_boxes:
                        state['is_statting'] = True
                        state['ratio_buffer'] = []
                        state['stat_frames_remaining'] = 5

                        min_distance = float('inf')
                        bucket_center_x = (state['last_dumping_bucket_xyxy'][0] + state['last_dumping_bucket_xyxy'][
                            2]) / 2

                        for truck in truck_boxes:
                            truck_center_x = (truck['xmin'] + truck['xmax']) / 2
                            distance = abs(truck_center_x - bucket_center_x)
                            if distance < min_distance:
                                min_distance = distance
                                state['current_truck_xyxy'] = [truck['xmin'], truck['ymin'], truck['xmax'],
                                                               truck['ymax']]

                        state['last_dumping_bucket_xyxy'] = None
                    else:
                        state['stable_frames_remaining'] = 1

            # === E. 面积比例计算与切车结算 ===
            if state['is_statting'] and state['stat_frames_remaining'] > 0:
                state['stat_frames_remaining'] -= 1

                if state['current_truck_xyxy'] is not None and truck_boxes:
                    min_distance = float('inf')
                    current_truck = None
                    prev_center_x = (state['current_truck_xyxy'][0] + state['current_truck_xyxy'][2]) / 2

                    for truck in truck_boxes:
                        truck_center_x = (truck['xmin'] + truck['xmax']) / 2
                        distance = abs(truck_center_x - prev_center_x)
                        if distance < min_distance:
                            min_distance = distance
                            current_truck = truck

                    if current_truck is not None:
                        state['current_truck_xyxy'] = [current_truck['xmin'], current_truck['ymin'],
                                                       current_truck['xmax'], current_truck['ymax']]
                        t_x1, t_y1, t_x2, t_y2 = state['current_truck_xyxy']
                        truck_area = (t_x2 - t_x1) * (t_y2 - t_y1)

                        if truck_area > 0:
                            mine_area = 0
                            for mine in mine_boxes:
                                m_xyxy = [mine['xmin'], mine['ymin'], mine['xmax'], mine['ymax']]
                                if check_horizontal_overlap(state['current_truck_xyxy'], m_xyxy):
                                    current_mine_area = (m_xyxy[2] - m_xyxy[0]) * (m_xyxy[3] - m_xyxy[1])
                                    if current_mine_area > mine_area:
                                        mine_area = current_mine_area
                            ratio = mine_area / truck_area
                        else:
                            ratio = 0.0
                    else:
                        ratio = 0.0
                else:
                    ratio = 0.0

                state['ratio_buffer'].append(ratio)

                if state['stat_frames_remaining'] == 0:
                    state['is_statting'] = False
                    avg_ratio = sum(state['ratio_buffer']) / len(state['ratio_buffer']) if state[
                        'ratio_buffer'] else 0.0

                    if state['last_avg_ratio'] is None:
                        state['last_avg_ratio'] = avg_ratio
                        if state['pending_buckets'] > 0:
                            state['current_truck_buckets'] += state['pending_buckets']
                            # 增加日志：避免第一次静默吞并
                            merge_msg = f"[STAT-MERGE] Initial confirmed. Loads: {state['current_truck_buckets']}"
                            state['event_logs'].insert(0, merge_msg)
                            print(f">>> {merge_msg}")
                            state['pending_buckets'] = 0
                    else:
                        if state['last_avg_ratio'] == 0.0 and avg_ratio == 0.0:
                            if state['pending_buckets'] > 0:
                                state['current_truck_buckets'] += state['pending_buckets']
                                merge_msg = f"[STAT-MERGE] Null overlap confirmed. Loads: {state['current_truck_buckets']}"
                                state['event_logs'].insert(0, merge_msg)
                                print(f">>> {merge_msg}")
                                state['pending_buckets'] = 0
                        else:
                            decline = (state['last_avg_ratio'] - avg_ratio) / state['last_avg_ratio'] if state[
                                                                                                             'last_avg_ratio'] > 0 else 0.0

                            if avg_ratio == 0.0 or decline >= MODEL_CONFIG['DECLINE_THRESH']:
                                force_complete_truck()

                                state['is_truck_active'] = True
                                state['total_truck_count'] += 1
                                state['ticket_id'] = f"TKT_{int(now * 1000)}"
                                state['current_truck_buckets'] = state['pending_buckets']
                                state['pending_buckets'] = 0
                                state['last_action_time'] = now
                                state['has_pushed_timeout'] = False

                                cut_msg = f"[CUT-TRUCK] New Truck {state['ticket_id']} inherited {state['current_truck_buckets']} bucket(s)."
                                state['event_logs'].insert(0, cut_msg)
                                print(f">>> {cut_msg}")
                            else:
                                if state['pending_buckets'] > 0:
                                    state['current_truck_buckets'] += state['pending_buckets']
                                    merge_msg = f"[STAT-MERGE] Same truck confirmed. Loads: {state['current_truck_buckets']}"
                                    state['event_logs'].insert(0, merge_msg)
                                    print(f">>> {merge_msg}")
                                    state['pending_buckets'] = 0

                        state['last_avg_ratio'] = avg_ratio

                    state['current_truck_xyxy'] = None
                    state['last_dumping_bucket_xyxy'] = None

            # === F. 快速强行合并兜底 ===
            if state['pending_buckets'] > 0 and state['frames_since_bucket_empty'] > 15:
                state['current_truck_buckets'] += state['pending_buckets']
                fb_msg = f"[FALLBACK] Auto-confirm {state['pending_buckets']} load(s) after 15 frames."
                state['event_logs'].insert(0, fb_msg)
                print(f">>> {fb_msg}")
                state['pending_buckets'] = 0

            # === 可视化渲染 ===
            for res in results:
                c_id = res['class_id']
                if c_id < len(MODEL_CONFIG['CLASSES']):
                    label = MODEL_CONFIG['CLASSES'][c_id]
                else:
                    label = f"unknown_{c_id}"

                color = CLASS_COLORS.get(label, (200, 200, 200))
                cv2.rectangle(frame, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), color, 2)
                text = f"{label}: {res['score']:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (res['xmin'], res['ymin'] - th - 5), (res['xmin'] + tw, res['ymin']), color, -1)
                cv2.putText(frame, text, (res['xmin'], res['ymin'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.rectangle(frame, (20, 20), (450, 150), (0, 0, 0), -1)
            cv2.putText(frame, f"Trucks: {state['total_truck_count']}", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"Cur Buckets: {state['current_truck_buckets']}", (35, 95), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {state['ticket_id']}", (35, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            output_canvas = np.zeros((orig_h, canvas_w, 3), dtype=np.uint8)
            output_canvas[:, :orig_w] = frame

            log_start_x = orig_w + 15
            cv2.putText(output_canvas, "--- Event Logs ---", (log_start_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            for i, log_text in enumerate(state['event_logs'][:20]):
                color = (0, 255, 255) if "[START]" in log_text \
                    else (0, 255, 0) if (
                            "[LOAD]" in log_text or "[FAST-RELEASE]" in log_text or "[FALLBACK]" in log_text or "[STAT-MERGE]" in log_text) \
                    else (255, 165, 0) if "[PENDING]" in log_text \
                    else (255, 0, 255) if "[CUT-TRUCK]" in log_text \
                    else (200, 200, 200) if "[DB PUSH]" in log_text \
                    else (0, 0, 255)

                cv2.putText(output_canvas, log_text, (log_start_x, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,
                            1)

            writer.write(output_canvas)

            if frame_count % 10 == 0:
                print(f"帧数: {frame_count:04d} | 推理耗时: {infer_time_ms:.2f} ms")

        force_complete_truck()

    except Exception as e:
        import traceback
        print(f"❌ 流水线运行时异常捕获:\n{traceback.format_exc()}")
    finally:
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'writer' in locals(): writer.release()
        rknn_yolo.release()
        print("--> 硬件资源安全回收。")


if __name__ == '__main__':
    run_inference()