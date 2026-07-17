#include "excavator_pipeline.h"
#include "rknn_api.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// ================= 配置与数据结构 =================
struct PipelineConfig {
    int yolo_input_w = 320;
    int yolo_input_h = 320;
    int yolo_reg_max = 16; // YOLO 的 DFL (Distribution Focal Loss) 参数

    float conf_thresh = 0.3f; // YOLO 置信度阈值
    float iou_thresh = 0.45f; // NMS 重叠阈值，用于去重
    int nms_offset = 4096; // 多类别 NMS 的偏移量技巧，让不同类别的框（如满斗和卸矿）在物理上绝对不重叠，从而分开计算 NMS

    long long timeout_ms = 60000; // 业务超时时间：1 分钟没动作就算超时
    float decline_thresh = 0.75f; // 切车阈值：矿石面积比例断崖式下跌 75% 时判定为换车
};

const std::vector<std::string> CLASSES = {"bucket-empty", "bucket-full", "truck", "loading", "dumping", "mine"};

struct BBox {
    int xmin, ymin, xmax, ymax;
    float score;
    int class_id;
};

// ================= 服务器事件结构 =================
struct BucketEvent {
    std::string ticket_id;
    int total_truck_count;
    int current_bucket_count;
    long long dump_start_time;
    long long dump_end_time;
    float last_mineral_ratio;
};

struct TruckEvent {
    std::string ticket_id;
    int total_truck_count;
    int total_bucket_count;
    long long load_start_time;
    long long load_end_time;
    int completed_type;
};

struct PendingBucket {
    long long dump_start_time;
    long long dump_end_time;
};

// ================= 状态机与缓存区 =================
struct PipelineState {
    std::string ticket_id = "WAITING"; // 当前作业的车次票号
    int total_truck_count = 0; // 历史总装车数
    int total_bucket_count = 0; // 历史总铲数
    int current_truck_buckets = 0; // 当前这辆车已经确认装进去的铲数

    int pending_buckets = 0; // 挂起铲数：已经发生动作，但还在等待 Dumping 结束或矿石比例核算的铲数
    int frames_since_bucket_empty = 0;
    std::vector<PendingBucket> pending_queue; // 记录挂起铲时间戳的队列

    bool has_pushed_timeout = false; // 防止超时事件被疯狂重复推送的锁
    int timeout_bucket_count = -1; // 记录超时瞬间的铲数，用于拦截假死车

    bool bucket_full = false; // 铲斗当前状态：true=满，false=空
    bool dumping_active = false; // 卸矿动作是否正在持续
    int dumping_frame_count = 0; // 卸矿动作维持的帧数
    int dumping_lost_frames = 0; // 容忍漏检的帧数（抗闪烁）

    int retry_count = 0;
    int max_retry_count = 5;

    long long current_dump_start_time = 0;
    long long truck_load_start_time = 0;
    long long truck_load_end_time = 0;
    long long last_dump_end_time = 0;
    long long last_action_time = 0;

    bool is_truck_active = false;

    bool pending_bucket_secured = false; // 延迟核销锁：空斗出现时置 true，等 Dumping 彻底结束后才去结算铲数
    long long secured_dump_start_time = 0;

    cv::Rect last_dumping_box = cv::Rect(0,0,0,0);
    cv::Rect current_truck_box = cv::Rect(0,0,0,0);
    cv::Rect last_dumping_bucket_box = cv::Rect(0,0,0,0);

    cv::Rect ui_bucket_box = cv::Rect(0,0,0,0);
    cv::Rect ui_truck_box = cv::Rect(0,0,0,0);
    std::vector<BBox> ui_all_detections;

    // 用于计算矿石比例（切车）的缓存
    int stable_frames_remaining = 0; // 卸矿结束后，等待画面稳定的帧数
    bool is_statting = false; // 是否正在进行 “矿石 / 卡车比值” 的 15 帧采样
    int stat_frames_remaining = 0; // 剩余采样帧数
    std::vector<float> ratio_buffer; // 存放这 15 帧里每一帧的比值
    float last_avg_ratio = -1.0f; // 上一铲的平均比值（用于和当前铲对比，找断崖下跌）

    std::vector<BucketEvent> pending_bucket_events;
    std::vector<TruckEvent> pending_truck_events;
};

class ExcavatorPipeline {
private:
    rknn_context rknn_yolo = 0;
    PipelineState state;
    PipelineConfig config;

    // 生成卡车票号
    static std::string generate_ticket_id() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        return "TKT_" + std::to_string(ms);
    }

    static long long get_current_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // 等比例缩放与填充
    cv::Mat letterbox(cv::Mat& img, float& ratio, int& dw, int& dh) const {
        int h = img.rows, w = img.cols;
        ratio = std::min((float)config.yolo_input_w / w, (float)config.yolo_input_h / h);
        int new_unpad_w = std::round(w * ratio);
        int new_unpad_h = std::round(h * ratio);
        dw = (config.yolo_input_w - new_unpad_w) / 2;
        dh = (config.yolo_input_h - new_unpad_h) / 2;
        cv::Mat resized, output;
        if (w != new_unpad_w || h != new_unpad_h) {
            resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img.clone();
        }
        copyMakeBorder(resized, output, dh, config.yolo_input_h - new_unpad_h - dh,
                           dw, config.yolo_input_w - new_unpad_w - dw,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return output;
    }

    // 检查两个 Box 是否在水平上存在重叠
    static bool check_horizontal_overlap(const cv::Rect& b1, const cv::Rect& b2) {
        return !(b1.x + b1.width < b2.x || b2.x + b2.width < b1.x);
    }

    // 强制完结当前卡车（0 换车，1 超时）
    void force_complete_truck(int completed_type = 0) {
        if (state.is_truck_active) {

            bool should_push_event = true;

            if (completed_type == 1) {
                // 如果是超时，拍下一张铲数快照
                state.timeout_bucket_count = state.current_truck_buckets;
            }
            else if (completed_type == 0) {
                // 如果是正常完结，对比现在的铲数和超时瞬间的快照
                // 如果一模一样，说明这辆老车超时后根本没有进账，直接拦截这条重复的完结事件
                if (state.timeout_bucket_count != -1 && state.timeout_bucket_count == state.current_truck_buckets) {
                    should_push_event = false;
                }
            }

            if (should_push_event) {
                TruckEvent te;
                te.ticket_id = state.ticket_id;
                te.total_truck_count = state.total_truck_count;
                te.total_bucket_count = state.current_truck_buckets;
                te.load_start_time = state.truck_load_start_time;
                te.completed_type = completed_type;

                long long end_time = state.truck_load_end_time > 0 ? state.truck_load_end_time : state.last_action_time;
                if (end_time <= 0) end_time = get_current_time_ms();
                te.load_end_time = end_time;

                state.pending_truck_events.push_back(te);
            }

            // 无论是否推送事件，一旦发生换车，状态机的清空动作照常执行
            if (completed_type == 0) {
                state.is_truck_active = false;
                state.ticket_id = "WAITING";
                state.current_truck_buckets = 0;
                state.has_pushed_timeout = false;
                state.timeout_bucket_count = -1;
                state.last_avg_ratio = -1.0f;
            }
        }
    }

    // 向 Android 业务层提交挂起的铲数
    void commit_pending_buckets() {
        for (const auto& pb : state.pending_queue) {
            state.current_truck_buckets++;
            state.truck_load_end_time = pb.dump_end_time;

            BucketEvent be;
            be.ticket_id = state.ticket_id;
            be.total_truck_count = state.total_truck_count;
            be.current_bucket_count = state.current_truck_buckets;
            be.dump_start_time = pb.dump_start_time;
            be.dump_end_time = pb.dump_end_time;

            // 只要是没算出来的负数，统统默认为 0.0f
            be.last_mineral_ratio = (state.last_avg_ratio < 0) ? 0.0f : state.last_avg_ratio;

            state.pending_bucket_events.push_back(be);
        }
        state.pending_buckets = 0;
        state.pending_queue.clear();
    }

    // 切车（发现比例断崖下跌时触发）
    void cut_truck(long long now, float new_ratio = 0.0f) {
        force_complete_truck(0);
        state.is_truck_active = true;
        state.total_truck_count++;
        state.ticket_id = generate_ticket_id();
        state.current_truck_buckets = 0;
        state.has_pushed_timeout = false; // 重计超时
        state.last_action_time = now;

        // 如果 pending_queue 里有数据，说明正是队列里的这一铲引发了切车
        if (!state.pending_queue.empty()) {
            state.truck_load_start_time = state.pending_queue[0].dump_start_time;
        } else {
            state.truck_load_start_time = now; // 极少发生的异常情况
        }

        // 赶在 commit_pending_buckets 前，把刚计算好的比例赋给新卡车，防止它用老车留下的高比例或初始值 -1 提交
        state.last_avg_ratio = new_ratio;

        commit_pending_buckets();
    }

public:
    ExcavatorPipeline(const void* yolo_data, const int yolo_size) {
        if (yolo_data && yolo_size > 0) {
            rknn_init(&rknn_yolo, const_cast<void *>(yolo_data), yolo_size, 0, nullptr);
            state.last_action_time = 0;
        }
    }

    ~ExcavatorPipeline() {
        if (rknn_yolo) rknn_destroy(rknn_yolo);
    }

    // 程序运行期间动态修改 YOLO 参数
    void updateConfig(const PipelineConfig& new_config) {
        long long old = config.timeout_ms;
        this->config = new_config;
        this->config.timeout_ms = old; // 防御性编程，更新这个参数用 setTimeout()
    }

    // 修改全局的超时判定时间
    void setTimeout(long long timeout_ms) {
        this->config.timeout_ms = timeout_ms;
    }

    //断电 / 崩溃热恢复
    void restoreState(const std::string& ticket_id, int bucket_count, float last_mineral_ratio) {
        state.ticket_id = ticket_id;
        state.current_truck_buckets = bucket_count;
        state.last_avg_ratio = last_mineral_ratio;

        if (bucket_count > 0) {
            state.is_truck_active = true;
            long long now = get_current_time_ms();
            state.truck_load_start_time = now;
            state.truck_load_end_time = now;
            state.last_dump_end_time = now;
            state.last_action_time = 0;
            state.timeout_bucket_count = -1;

            if (state.total_truck_count == 0) {
                state.total_truck_count = 1;
            }
        }
    }

    // 预留设置总车数方法
    void setTotalTruckCount(int count) {
        if (count >= 0) state.total_truck_count = count;
    }

    void clear_events() {
        state.pending_bucket_events.clear();
        state.pending_truck_events.clear();
    }

    PipelineState& getState() {
        return state;
    }

    // 超时判定 -> 图像预处理 -> NPU 模型推理 -> YOLO 层硬解码 -> 空间物理约束判定 -> 限制性状态机切换 -> 矿石比例计算 -> UI 数据刷新
    void process(cv::Mat& frame) {
        long long now = get_current_time_ms();
        state.frames_since_bucket_empty++;

        // 设置了有效时间戳 && (卡车里已经有矿或有小铲挂起) && 距离上一次动作超过了配置的超时阈值
        if (state.last_action_time > 0 && (state.current_truck_buckets > 0 || state.pending_buckets > 0) && (now - state.last_action_time > config.timeout_ms)) {
            if (!state.has_pushed_timeout) { // 状态锁，确保单次超时事件只推送一次
                state.has_pushed_timeout = true;
                force_complete_truck(1); // 触发超时完结老车
            }
        }

        float ratio; int dw, dh;
        cv::Mat prep_img = letterbox(frame, ratio, dw, dh);
        cv::cvtColor(prep_img, prep_img, cv::COLOR_BGR2RGB); // OpenCV 默认 BGR，RKNN 默认 RGB

        rknn_input inputs[1]; memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0; inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = config.yolo_input_w * config.yolo_input_h * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC; inputs[0].buf = prep_img.data;
        rknn_inputs_set(rknn_yolo, 1, inputs);

        rknn_run(rknn_yolo, NULL);

        rknn_output yolo_outputs[3]; memset(yolo_outputs, 0, sizeof(yolo_outputs));
        for (int i = 0; i < 3; ++i) yolo_outputs[i].want_float = 1;
        rknn_outputs_get(rknn_yolo, 3, yolo_outputs, NULL);

        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;
        int strides[3] = {8, 16, 32};
        int num_classes = CLASSES.size();

        for (int i = 0; i < 3; ++i) {
            int grid_w = config.yolo_input_w / strides[i];
            int grid_h = config.yolo_input_h / strides[i];
            float* out_ptr = (float*)yolo_outputs[i].buf;
            int map_size = grid_h * grid_w;

            for (int h = 0; h < grid_h; ++h) {
                for (int w = 0; w < grid_w; ++w) {
                    int spatial_idx = h * grid_w + w;
                    float max_score = -1.0f;
                    int best_class = -1;

                    for (int c = 0; c < num_classes; ++c) {
                        int c_idx = (4 * config.yolo_reg_max + c) * map_size + spatial_idx;
                        float raw_val = out_ptr[c_idx];
                        raw_val = std::max(-88.0f, std::min(88.0f, raw_val));
                        float score = 1.0f / (1.0f + std::exp(-raw_val));
                        if (score > max_score) { max_score = score; best_class = c; }
                    }

                    if (max_score > config.conf_thresh) {
                        float dfl_preds[4];
                        for (int k = 0; k < 4; ++k) {
                            float max_reg = -1e9f; std::vector<float> reg_raw(config.yolo_reg_max);
                            for (int r = 0; r < config.yolo_reg_max; ++r) {
                                float val = out_ptr[(k * config.yolo_reg_max + r) * map_size + spatial_idx];
                                reg_raw[r] = val; if (val > max_reg) max_reg = val;
                            }
                            float sum_exp = 0.0f; float dfl_val = 0.0f;
                            for (int r = 0; r < config.yolo_reg_max; ++r) {
                                reg_raw[r] = std::exp(reg_raw[r] - max_reg); sum_exp += reg_raw[r];
                            }
                            for (int r = 0; r < config.yolo_reg_max; ++r) dfl_val += (reg_raw[r] / sum_exp) * r;
                            dfl_preds[k] = dfl_val;
                        }

                        float cx = (w + 0.5f - dfl_preds[0]) * strides[i];
                        float cy = (h + 0.5f - dfl_preds[1]) * strides[i];
                        float x2 = (w + 0.5f + dfl_preds[2]) * strides[i];
                        float y2 = (h + 0.5f + dfl_preds[3]) * strides[i];

                        int orig_xmin = std::round((cx - dw) / ratio);
                        int orig_ymin = std::round((cy - dh) / ratio);
                        int orig_xmax = std::round((x2 - dw) / ratio);
                        int orig_ymax = std::round((y2 - dh) / ratio);

                        nms_boxes.push_back(cv::Rect(orig_xmin, orig_ymin, orig_xmax - orig_xmin, orig_ymax - orig_ymin));
                        nms_scores.push_back(max_score);
                        nms_class_ids.push_back(best_class);
                    }
                }
            }
        }
        rknn_outputs_release(rknn_yolo, 3, yolo_outputs); // 释放 RKNN 的底层硬件缓冲区指针，防止内存雪崩

        std::vector<int> indices;
        std::vector<cv::Rect> offset_boxes;
        for (size_t i = 0; i < nms_boxes.size(); ++i) {
            offset_boxes.push_back(cv::Rect(nms_boxes[i].x + nms_class_ids[i] * config.nms_offset,
                nms_boxes[i].y + nms_class_ids[i] * config.nms_offset, nms_boxes[i].width, nms_boxes[i].height));
        }
        cv::dnn::NMSBoxes(offset_boxes, nms_scores, config.conf_thresh, config.iou_thresh, indices);

        std::vector<BBox> truck_boxes, bucket_boxes, dumping_boxes, mine_boxes;

        state.ui_all_detections.clear(); // 清洗界面上一帧的所有历史脏框

        // 将通过 NMS 筛查的框分类归档到各自的 Vector 数组中
        for (int idx : indices) {
            BBox box = {nms_boxes[idx].x, nms_boxes[idx].y, nms_boxes[idx].x + nms_boxes[idx].width, nms_boxes[idx].y + nms_boxes[idx].height, nms_scores[idx], nms_class_ids[idx]};
            state.ui_all_detections.push_back(box);

            if (box.class_id == 2) truck_boxes.push_back(box);
            else if (box.class_id == 0 || box.class_id == 1) bucket_boxes.push_back(box);
            else if (box.class_id == 4) dumping_boxes.push_back(box);
            else if (box.class_id == 5) mine_boxes.push_back(box);
        }

        // 如果当前状态机认为铲斗根本不是满的，且没有进入锁死期，直接干掉这一帧的 dumping 框
        // 目的：过滤掉挖掘机在矿山背景摆臂、大风吹拂烟尘产生的偶尔单帧假 dumping 误检
        if (!state.bucket_full && !state.pending_bucket_secured) {
            dumping_boxes.clear();
        }

        // ============================== B. 铲斗满空判定与解锁 ==============================
        if (!bucket_boxes.empty()) {
            // B.1. 选出当前帧置信度分数最高的那只铲斗
            auto best_bucket = bucket_boxes[0];
            for (const auto& bx : bucket_boxes) if (bx.score > best_bucket.score) best_bucket = bx;
            cv::Rect bb_rect(best_bucket.xmin, best_bucket.ymin, best_bucket.xmax - best_bucket.xmin, best_bucket.ymax - best_bucket.ymin);

            // B.2. 锁定现场正在装载的唯一主卡车
            cv::Rect main_truck_rect(0,0,0,0);
            if (!truck_boxes.empty()) {
                auto best_t = truck_boxes[0];
                float max_area = (best_t.xmax - best_t.xmin) * (best_t.ymax - best_t.ymin);
                for (const auto& t : truck_boxes) { // 挑出画面里占地面积最大的车
                    float area = (t.xmax - t.xmin) * (t.ymax - t.ymin);
                    if (area > max_area) { max_area = area; best_t = t; }
                }
                main_truck_rect = cv::Rect(best_t.xmin, best_t.ymin, best_t.xmax - best_t.xmin, best_t.ymax - best_t.ymin);
            }

            // B.3. 状态分流
            if (best_bucket.class_id == 1) { // 检测到满斗
                // 如果上一次动作挂起还没核销，且状态稳定，立刻在进满斗的瞬间把上一铲存进大车
                if (state.pending_buckets > 0 && !state.is_statting && !state.dumping_active) {
                    commit_pending_buckets();
                }
                if (!state.bucket_full) { // 历史状态是空校验

                    bool overlap_main = false;
                    bool is_digging_low = false; // 高度拦截变量

                    if (main_truck_rect.area() > 0) {
                        overlap_main = check_horizontal_overlap(bb_rect, main_truck_rect); // 检查左右 X 轴是否重合

                        // 如果铲斗的上边缘（Y值），掉到了卡车最高身位的 40% 以下（朝向地面靠拢）
                        // 证明挖掘机一定是在地势低的坑里挖土，绝对不可能在卡车头顶上变成满斗
                        if (bb_rect.y > main_truck_rect.y + main_truck_rect.height * 0.4f) {
                            is_digging_low = true;
                        }
                    }

                    // 转换放行门槛：铲斗和卡车左右没重叠（在车厢外挖掘），或者虽然在车厢投影内，但在极低的地面位置挖矿
                    // 同时画面上不能有 dumping 卸矿残影。满足这些苛刻的物理条件，才准许将全局状态反转为 “满斗”
                    if ((!overlap_main || is_digging_low) && !state.dumping_active && dumping_boxes.empty()) {
                        state.bucket_full = true;
                    }
                }
            } else if (best_bucket.class_id == 0) { // 检测到空斗
                if (state.bucket_full) { // 必须之前是满斗，才能往下走

                    bool overlap_main = false;
                    bool is_hovering_high = true; // 高空卸矿判定器

                    if (main_truck_rect.area() > 0) {
                        overlap_main = check_horizontal_overlap(bb_rect, main_truck_rect); // 必须在卡车投影里

                        // 如果空斗在靠地面的地方被识别，绝对是甩臂残影或者扬尘误检，高空拦截置 false
                        if (bb_rect.y > main_truck_rect.y + main_truck_rect.height * 0.4f) {
                            is_hovering_high = false;
                        }
                    }

                    // 铲斗开到了卡车上空（左右重叠） && 必须悬停在高空位置
                    if (overlap_main && is_hovering_high) {
                        state.bucket_full = false; // 满斗卸空，状态机状态反转

                        // 如果当前画面正好抓到了 dumping 状态，说明烟尘很大，延迟核销战略
                        if (state.dumping_active || !dumping_boxes.empty()) {
                            state.pending_bucket_secured = true; // 上一把状态锁，通知后续流程等倒完再记账
                            // 追溯卸矿起点时间戳
                            state.secured_dump_start_time = state.current_dump_start_time > 0 ? state.current_dump_start_time : now;
                        } else {
                            // 没 dumping，常规记账
                            state.total_bucket_count++;
                            state.pending_buckets++;
                            state.frames_since_bucket_empty = 0;
                            state.last_action_time = now;
                            state.has_pushed_timeout = false;

                            if (!state.is_truck_active) {
                                state.is_truck_active = true;
                                state.total_truck_count++;
                                state.ticket_id = generate_ticket_id();
                                state.current_truck_buckets = 0;
                                state.truck_load_start_time = state.current_dump_start_time > 0 ? state.current_dump_start_time : now;
                            }

                            PendingBucket pb;
                            pb.dump_start_time = state.current_dump_start_time > 0 ? state.current_dump_start_time : now;
                            if (now - pb.dump_start_time < 500) pb.dump_start_time = now - 1500;
                            pb.dump_end_time = now;
                            state.pending_queue.push_back(pb);
                            state.current_dump_start_time = 0;

                            if (state.stable_frames_remaining == 0 && !state.is_statting) {
                                state.stable_frames_remaining = 1;
                            }
                        }
                    }
                }
            }
        }

        // ============================== C. 跟踪 Dumping 状态 ==============================
        bool has_dumping = !dumping_boxes.empty();

        if (has_dumping) { // 画面上有卸矿的动作
            state.dumping_frame_count++;
            state.dumping_lost_frames = 0; // 只要能看到，丢失容忍计数直接归零

            if (state.dumping_frame_count == 1) state.current_dump_start_time = now; // 记录发生动作的开始时间点

            if (!state.dumping_active && state.dumping_frame_count >= 2) {
                state.dumping_active = true; // 连续 2 帧稳定目击，确定真正的卸矿行为拉开序幕
            }

            // 如果正在倒矿，说明画面的卡车和矿石都在剧烈变化，立刻强行阻断、清除后续的比例计算计数器
            if (state.stable_frames_remaining > 0 || state.is_statting) {
                state.stable_frames_remaining = 0;
                state.is_statting = false;
                state.stat_frames_remaining = 0;
                state.ratio_buffer.clear();
                state.current_truck_box = cv::Rect(0,0,0,0);
                state.last_dumping_bucket_box = cv::Rect(0,0,0,0);
            }
        } else { // 画面上突然看不见 dumping 框了
            if (state.dumping_active) {
                state.dumping_lost_frames++; // 触发抗闪烁计数器：没准是被烟雾挡了呢？先加 1 帧观察下

                // 只有当连续 6 帧完全看不到 dumping 时，算法才敢断定——这一铲矿彻底倒完了！
                if (state.dumping_lost_frames > 6) {
                    state.dumping_active = false;
                    state.dumping_frame_count = 0;
                    state.dumping_lost_frames = 0;


                    if (state.pending_bucket_secured) { // 检查刚才在逻辑块B里上的那把安全锁
                        state.pending_bucket_secured = false;

                        // 启动正式记账：增补全盘铲数，塞入记账队列
                        state.total_bucket_count++;
                        state.pending_buckets++;
                        state.frames_since_bucket_empty = 0;
                        state.last_action_time = now;
                        state.has_pushed_timeout = false;

                        if (!state.is_truck_active) {
                            state.is_truck_active = true;
                            state.total_truck_count++;
                            state.ticket_id = generate_ticket_id();
                            state.current_truck_buckets = 0;
                            state.truck_load_start_time = state.secured_dump_start_time > 0 ? state.secured_dump_start_time : now;
                        }

                        // 打包这次延迟结算的单铲，注入精准的 dump_end_time 结束时间（即当前时间 now）
                        PendingBucket pb;
                        pb.dump_start_time = state.secured_dump_start_time > 0 ? state.secured_dump_start_time : now;
                        pb.dump_end_time = now;
                        state.pending_queue.push_back(pb);
                        state.current_dump_start_time = 0;
                    }

                    state.stable_frames_remaining = 1;

                    if (!bucket_boxes.empty()) {
                        auto b = bucket_boxes[0];
                        for (const auto& bx : bucket_boxes) if (bx.score > b.score) b = bx;
                        state.last_dumping_bucket_box = cv::Rect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin);
                    } else {
                        state.last_dumping_bucket_box = cv::Rect(0,0,0,0);
                    }
                }
            } else {
                state.dumping_frame_count = 0;
                state.dumping_lost_frames = 0;
            }
        }

        // ============================== D. 寻找用于计算比值的卡车 ==============================
        if (state.stable_frames_remaining > 0) {
            state.stable_frames_remaining--;
            if (state.stable_frames_remaining == 0) { // 尘埃落定，倒计时归零

                // 如果刚才没抓到铲斗坐标，从当前帧强行挑一只出来作为几何参照物
                if (state.last_dumping_bucket_box.area() == 0 && !bucket_boxes.empty()) {
                    auto b = bucket_boxes[0];
                    for (const auto& bx : bucket_boxes) if (bx.score > b.score) b = bx;
                    state.last_dumping_bucket_box = cv::Rect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin);
                }

                if (state.last_dumping_bucket_box.area() > 0) {
                    if (!truck_boxes.empty()) {
                        state.is_statting = true;
                        state.ratio_buffer.clear(); // 清空 15 帧比例池
                        state.stat_frames_remaining = 15; // 连续采样 15 帧
                        state.retry_count = 0;

                        float min_dist = 1e9;
                        float b_cx = state.last_dumping_bucket_box.x + state.last_dumping_bucket_box.width / 2.0f;

                        // 找到那个和刚刚倒矿的铲斗水平 X 轴距离最近的卡车厢
                        for (const auto& t : truck_boxes) {
                            float t_cx = t.xmin + (t.xmax - t.xmin) / 2.0f;
                            float dist = std::abs(t_cx - b_cx);
                            if (dist < min_dist) {
                                min_dist = dist;
                                state.current_truck_box = cv::Rect(t.xmin, t.ymin, t.xmax - t.xmin, t.ymax - t.ymin);
                            }
                        }
                        state.last_dumping_bucket_box = cv::Rect(0,0,0,0);
                    } else {
                        state.retry_count++;
                        if (state.retry_count < state.max_retry_count) {
                            state.stable_frames_remaining = 1;
                        } else {
                            // 找车 5 次彻底失败，不要挂机，强行转 0 结算，防止遗留的 -1 泄露
                            state.retry_count = 0;
                            state.last_dumping_bucket_box = cv::Rect(0,0,0,0);

                            if (state.last_avg_ratio < 0) state.last_avg_ratio = 0.0f;
                            if (state.pending_buckets > 0) commit_pending_buckets();
                        }
                    }
                } else {
                    state.retry_count++;
                    if (state.retry_count < state.max_retry_count) {
                        state.stable_frames_remaining = 1;
                    } else {
                        // 找车 5 次彻底失败，不要挂机，强行转 0 结算，防止遗留的 -1 泄露
                        state.retry_count = 0;
                        state.last_dumping_bucket_box = cv::Rect(0,0,0,0);

                        if (state.last_avg_ratio < 0) state.last_avg_ratio = 0.0f;
                        if (state.pending_buckets > 0) commit_pending_buckets();
                    }
                }
            }
        }

        // ============================== E. 真实比例断崖下跌计算 ==============================
        if (state.is_statting && state.stat_frames_remaining > 0) {
            state.stat_frames_remaining--;

            if (state.current_truck_box.area() > 0 && !truck_boxes.empty()) {
                float min_dist = 1e9;
                cv::Rect best_t(0,0,0,0);
                float prev_cx = state.current_truck_box.x + state.current_truck_box.width / 2.0f;

                // 在当前帧寻找上一帧被锁死的那辆车的几何继承者（防止卡车小范围移动或框抖动）
                for (const auto& t : truck_boxes) {
                    float t_cx = t.xmin + (t.xmax - t.xmin) / 2.0f;
                    float dist = std::abs(t_cx - prev_cx);
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_t = cv::Rect(t.xmin, t.ymin, t.xmax - t.xmin, t.ymax - t.ymin);
                    }
                }

                if (best_t.area() > 0) {
                    state.current_truck_box = best_t;
                    float truck_area = best_t.area();
                    float max_mine_area = 0;
                    // 搜寻并过滤出完全落在这辆卡车车厢投影范围内的最大矿石（mine）检测面积
                    for (const auto& m : mine_boxes) {
                        cv::Rect m_rect(m.xmin, m.ymin, m.xmax - m.xmin, m.ymax - m.ymin);
                        if (check_horizontal_overlap(best_t, m_rect)) {
                            float m_area = m_rect.area();
                            if (m_area > max_mine_area) max_mine_area = m_area;
                        }
                    }
                    state.ratio_buffer.push_back(max_mine_area / truck_area);
                } else {
                    state.ratio_buffer.push_back(0.0f);
                }
            } else {
                state.ratio_buffer.push_back(0.0f);
            }

            // 15 帧采样窗口关闭，计算最终比例
            if (state.stat_frames_remaining == 0) {
                state.is_statting = false;

                // 只计算大于 0.0f 的有效面积。直接从均值计算中剔除因漏检造成的伪 0 帧
                float sum = 0;
                int valid_count = 0;
                for (float r : state.ratio_buffer) {
                    if (r > 0.0f) {  // 直接过滤掉漏检导致的 0
                        sum += r;
                        valid_count++;
                    }
                }

                // 只要这 15 帧内哪怕只有 1 帧检出了矿，就不会被误判为 0
                float avg_ratio = (valid_count > 0) ? (sum / valid_count) : 0.0f;

                if (state.last_avg_ratio < 0) { // 如果是今天开机的第一铲
                    state.last_avg_ratio = avg_ratio;
                    if (state.pending_buckets > 0) commit_pending_buckets();
                } else {
                    if (state.last_avg_ratio == 0.0f && avg_ratio == 0.0f) { // 两把都是空车
                        state.last_avg_ratio = avg_ratio;
                        if (state.pending_buckets > 0) commit_pending_buckets();
                    } else {
                        // 数学比例跌幅计算公式
                        float decline = state.last_avg_ratio > 0 ? (state.last_avg_ratio - avg_ratio) / state.last_avg_ratio : 0.0f;
                        if (avg_ratio == 0.0f || decline >= config.decline_thresh) {
                            cut_truck(now, avg_ratio);
                        } else {
                            // 比例正常，判定为同一辆卡车在续装，放行核销，铲数归老车
                            state.last_avg_ratio = avg_ratio;
                            if (state.pending_buckets > 0) commit_pending_buckets();
                        }
                    }
                }
                state.current_truck_box = cv::Rect(0,0,0,0); // 清除当前锁定车辆的卡点锁
            }
        }

        // ============================== F. 快速强行合并兜底 ==============================
        // 如果数据被挂起（pending_buckets > 0），但因为现场某些极端突发状况，导致长达 60 帧（大约2~3秒）
        // 既没有触发正常的 dumping，也没有触发比例统计。说明状态机卡在某个夹缝里了
        // 为了不漏计铲数，启动强行熔断机制，强行把比例洗成 0 并强制转正核销
        if (state.pending_buckets > 0 && state.frames_since_bucket_empty > 60 && !state.is_statting && !state.dumping_active) {
            if (state.last_avg_ratio < 0) state.last_avg_ratio = 0.0f;
            commit_pending_buckets();
        }

        // ============================== G. 更新 UI 实时渲染框 ==============================
        // 找出当前置信度最高的一只铲斗和面积最大的一辆卡车，塞进 ui_bucket_box 和 ui_truck_box
        // 保证外层 Android JNI 能够用极细的帧率在平板屏幕上流畅渲染出“绿色追踪框”，方便司机肉眼对账
        // 注：实际上一般都用全量画框，这个环节暂时保留
        state.ui_bucket_box = cv::Rect(0,0,0,0);
        if (!bucket_boxes.empty()) {
            auto b = bucket_boxes[0];
            for (const auto& bx : bucket_boxes) if (bx.score > b.score) b = bx;
            state.ui_bucket_box = cv::Rect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin);
        }

        state.ui_truck_box = cv::Rect(0,0,0,0);
        if (!truck_boxes.empty()) {
            auto best_t = truck_boxes[0];
            float max_area = (best_t.xmax - best_t.xmin) * (best_t.ymax - best_t.ymin);
            for (const auto& tx : truck_boxes) {
                float area = (tx.xmax - tx.xmin) * (tx.ymax - tx.ymin);
                if (area > max_area) {
                    max_area = area;
                    best_t = tx;
                }
            }
            state.ui_truck_box = cv::Rect(best_t.xmin, best_t.ymin, best_t.xmax - best_t.xmin, best_t.ymax - best_t.ymin);
        }
    }
};

extern "C" {
    void* init_pipeline_from_memory(const void* yolo_data, const int yolo_size) { return new ExcavatorPipeline(yolo_data, yolo_size); }
    void process_frame(void* handle, unsigned char* img_data, int width, int height, int channels) {
        if (!handle || !img_data) return;
        int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
        cv::Mat frame(height, width, type, img_data);
        ((ExcavatorPipeline*)handle)->process(frame);
    }
    void* get_pipeline_state(void* handle) { return handle ? (void*)&(((ExcavatorPipeline*)handle)->getState()) : nullptr; }
    void update_pipeline_config(void* handle, float conf_thresh, float iou_thresh) {
        if (!handle) return;
        PipelineConfig cfg; cfg.conf_thresh = conf_thresh; cfg.iou_thresh = iou_thresh;
        ((ExcavatorPipeline*)handle)->updateConfig(cfg);
    }
    void release_pipeline(void* handle) { if (handle) delete (ExcavatorPipeline*)handle; }
    void restore_pipeline_state(void* handle, const char* ticket_id, int bucket_count, float last_mineral_ratio) {
        if (handle) {
            std::string t_id = ticket_id ? std::string(ticket_id) : "";
            ((ExcavatorPipeline*)handle)->restoreState(t_id, bucket_count, last_mineral_ratio);
        }
    }
    void set_total_truck_count(void* handle, int count) {
        if (handle) ((ExcavatorPipeline*)handle)->setTotalTruckCount(count);
    }
    void set_pipeline_timeout(void* handle, long long timeout_ms) { if (handle) ((ExcavatorPipeline*)handle)->setTimeout(timeout_ms); }
    void clear_pipeline_events(void* handle) { if (handle) ((ExcavatorPipeline*)handle)->clear_events(); }
}