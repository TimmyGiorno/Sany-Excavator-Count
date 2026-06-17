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
    int yolo_reg_max = 16;

    float conf_thresh = 0.3f;
    float iou_thresh = 0.45f;
    int nms_offset = 4096;

    long long timeout_ms = 60000;
    float decline_thresh = 0.75f;
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
    std::string ticket_id = "WAITING";
    int total_truck_count = 0;
    int total_bucket_count = 0;
    int current_truck_buckets = 0;

    int pending_buckets = 0;
    int frames_since_bucket_empty = 0;
    std::vector<PendingBucket> pending_queue;

    bool has_pushed_timeout = false;

    bool bucket_full = false;
    bool dumping_active = false;
    int dumping_frame_count = 0;

    // 【新增】重试机制相关变量
    int retry_count = 0;
    int max_retry_count = 5;

    long long current_dump_start_time = 0;
    long long truck_load_start_time = 0;
    long long truck_load_end_time = 0;
    long long last_dump_end_time = 0;
    long long last_action_time = 0;

    bool is_truck_active = false;

    cv::Rect last_dumping_box = cv::Rect(0,0,0,0);
    cv::Rect current_truck_box = cv::Rect(0,0,0,0);
    cv::Rect last_dumping_bucket_box = cv::Rect(0,0,0,0);

    cv::Rect ui_bucket_box = cv::Rect(0,0,0,0);
    cv::Rect ui_truck_box = cv::Rect(0,0,0,0);
    std::vector<BBox> ui_all_detections;

    int stable_frames_remaining = 0;
    bool is_statting = false;
    int stat_frames_remaining = 0;
    std::vector<float> ratio_buffer;
    float last_avg_ratio = -1.0f;

    std::vector<BucketEvent> pending_bucket_events;
    std::vector<TruckEvent> pending_truck_events;
};

class ExcavatorPipeline {
private:
    rknn_context rknn_yolo = 0;
    PipelineState state;
    PipelineConfig config;

    static std::string generate_ticket_id() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        return "TKT_" + std::to_string(ms);
    }

    static long long get_current_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    cv::Mat letterbox(cv::Mat& img, float& ratio, int& dw, int& dh) {
        int h = img.rows, w = img.cols;
        ratio = std::min((float)config.yolo_input_w / w, (float)config.yolo_input_h / h);
        int new_unpad_w = std::round(w * ratio);
        int new_unpad_h = std::round(h * ratio);
        dw = (config.yolo_input_w - new_unpad_w) / 2;
        dh = (config.yolo_input_h - new_unpad_h) / 2;
        cv::Mat resized, output;
        if (w != new_unpad_w || h != new_unpad_h) {
            cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img.clone();
        }
        cv::copyMakeBorder(resized, output, dh, config.yolo_input_h - new_unpad_h - dh,
                           dw, config.yolo_input_w - new_unpad_w - dw,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return output;
    }

    static bool check_horizontal_overlap(const cv::Rect& b1, const cv::Rect& b2) {
        return !(b1.x + b1.width < b2.x || b2.x + b2.width < b1.x);
    }

    void force_complete_truck(int completed_type = 0) {
        if (state.is_truck_active) {
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

            state.is_truck_active = false;
            state.ticket_id = "WAITING";
            state.current_truck_buckets = 0;
            state.has_pushed_timeout = false;
            state.last_avg_ratio = -1.0f;
        }
    }

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
            be.last_mineral_ratio = state.last_avg_ratio;

            state.pending_bucket_events.push_back(be);
        }
        state.pending_buckets = 0;
        state.pending_queue.clear();
    }

    void cut_truck(long long now) {
        force_complete_truck(0);
        state.is_truck_active = true;
        state.total_truck_count++;
        state.ticket_id = generate_ticket_id();
        state.current_truck_buckets = 0;
        state.has_pushed_timeout = false;
        state.last_action_time = now;

        if (!state.pending_queue.empty()) {
            state.truck_load_start_time = state.pending_queue[0].dump_start_time;
        } else {
            state.truck_load_start_time = now;
        }
        commit_pending_buckets();
    }

public:
    ExcavatorPipeline(const void* yolo_data, const int yolo_size) {
        if (yolo_data && yolo_size > 0) {
            rknn_init(&rknn_yolo, const_cast<void *>(yolo_data), yolo_size, 0, nullptr);
            state.last_action_time = get_current_time_ms();
        }
    }

    ~ExcavatorPipeline() { if (rknn_yolo) rknn_destroy(rknn_yolo); }
    void updateConfig(const PipelineConfig& new_config) { long long old = config.timeout_ms; this->config = new_config; this->config.timeout_ms = old; }
    void setTimeout(long long timeout_ms) { this->config.timeout_ms = timeout_ms; }

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
            state.last_action_time = now;

            if (state.total_truck_count == 0) {
                state.total_truck_count = 1;
            }
        }
    }

    void clear_events() {
        state.pending_bucket_events.clear();
        state.pending_truck_events.clear();
    }

    PipelineState& getState() { return state; }

    void process(cv::Mat& frame) {
        long long now = get_current_time_ms();
        state.frames_since_bucket_empty++;

        if ((state.current_truck_buckets > 0 || state.pending_buckets > 0) && (now - state.last_action_time > config.timeout_ms)) {
            if (!state.has_pushed_timeout) {
                state.has_pushed_timeout = true;
                force_complete_truck(1);
            }
        }

        float ratio; int dw, dh;
        cv::Mat prep_img = letterbox(frame, ratio, dw, dh);
        cv::cvtColor(prep_img, prep_img, cv::COLOR_BGR2RGB);

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
        rknn_outputs_release(rknn_yolo, 3, yolo_outputs);

        std::vector<int> indices;
        std::vector<cv::Rect> offset_boxes;
        for (size_t i = 0; i < nms_boxes.size(); ++i) {
            offset_boxes.push_back(cv::Rect(nms_boxes[i].x + nms_class_ids[i] * config.nms_offset,
                nms_boxes[i].y + nms_class_ids[i] * config.nms_offset, nms_boxes[i].width, nms_boxes[i].height));
        }
        cv::dnn::NMSBoxes(offset_boxes, nms_scores, config.conf_thresh, config.iou_thresh, indices);

        std::vector<BBox> truck_boxes, bucket_boxes, dumping_boxes, mine_boxes;

        state.ui_all_detections.clear();

        for (int idx : indices) {
            BBox box = {nms_boxes[idx].x, nms_boxes[idx].y, nms_boxes[idx].x + nms_boxes[idx].width, nms_boxes[idx].y + nms_boxes[idx].height, nms_scores[idx], nms_class_ids[idx]};
            state.ui_all_detections.push_back(box);

            if (box.class_id == 2) truck_boxes.push_back(box);
            else if (box.class_id == 0 || box.class_id == 1) bucket_boxes.push_back(box);
            else if (box.class_id == 4) dumping_boxes.push_back(box);
            else if (box.class_id == 5) mine_boxes.push_back(box);
        }

        // === B. 铲斗满空判定与解锁 ===
        if (!bucket_boxes.empty()) {
            auto best_bucket = bucket_boxes[0];
            for (const auto& bx : bucket_boxes) if (bx.score > best_bucket.score) best_bucket = bx;
            cv::Rect bb_rect(best_bucket.xmin, best_bucket.ymin, best_bucket.xmax - best_bucket.xmin, best_bucket.ymax - best_bucket.ymin);

            if (best_bucket.class_id == 1) {
                if (state.pending_buckets > 0) commit_pending_buckets();
                if (!state.bucket_full) {
                    bool overlap = false;
                    for (auto& t : truck_boxes) {
                        if (check_horizontal_overlap(bb_rect, cv::Rect(t.xmin, t.ymin, t.xmax-t.xmin, t.ymax-t.ymin))) { overlap = true; break; }
                    }
                    if (!overlap) state.bucket_full = true;
                }
            } else if (best_bucket.class_id == 0) {
                if (state.bucket_full) {
                    bool overlap = false;
                    for (auto& t : truck_boxes) {
                        if (check_horizontal_overlap(bb_rect, cv::Rect(t.xmin, t.ymin, t.xmax-t.xmin, t.ymax-t.ymin))) { overlap = true; break; }
                    }
                    if (overlap) {
                        state.bucket_full = false;
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
                        pb.dump_end_time = now;
                        state.pending_queue.push_back(pb);
                    } else {
                        state.bucket_full = false;
                    }
                }
            }
        }

        // === C. 跟踪 Dumping 状态 ===
        bool has_dumping = !dumping_boxes.empty();

        if (has_dumping) {
            state.dumping_frame_count++;

            if (state.dumping_frame_count == 1) state.current_dump_start_time = now;

            // 【修改】阈值改为 5
            if (!state.dumping_active && state.dumping_frame_count >= 5) {
                state.dumping_active = true;
            }

            if (state.stable_frames_remaining > 0 || state.is_statting) {
                state.stable_frames_remaining = 0;
                state.is_statting = false;
                state.stat_frames_remaining = 0;
                state.ratio_buffer.clear();
                state.current_truck_box = cv::Rect(0,0,0,0);
                state.last_dumping_bucket_box = cv::Rect(0,0,0,0);
            }
        } else {
            state.dumping_frame_count = 0;
            if (state.dumping_active) {
                state.dumping_active = false;
                state.stable_frames_remaining = 1;

                if (!bucket_boxes.empty()) {
                    auto b = bucket_boxes[0];
                    for (const auto& bx : bucket_boxes) if (bx.score > b.score) b = bx;
                    state.last_dumping_bucket_box = cv::Rect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin);
                } else {
                    state.last_dumping_bucket_box = cv::Rect(0,0,0,0);
                }
            }
        }

        // === D. 寻找用于计算比值的卡车 (带重试机制) ===
        if (state.stable_frames_remaining > 0) {
            state.stable_frames_remaining--;
            if (state.stable_frames_remaining == 0) {

                if (state.last_dumping_bucket_box.area() == 0 && !bucket_boxes.empty()) {
                    auto b = bucket_boxes[0];
                    for (const auto& bx : bucket_boxes) if (bx.score > b.score) b = bx;
                    state.last_dumping_bucket_box = cv::Rect(b.xmin, b.ymin, b.xmax - b.xmin, b.ymax - b.ymin);
                }

                if (state.last_dumping_bucket_box.area() > 0) {
                    if (!truck_boxes.empty()) {
                        state.is_statting = true;
                        state.ratio_buffer.clear();
                        state.stat_frames_remaining = 10; // 【修改】采样窗口改为 10 帧
                        state.retry_count = 0; // 重置重试

                        float min_dist = 1e9;
                        float b_cx = state.last_dumping_bucket_box.x + state.last_dumping_bucket_box.width / 2.0f;

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
                        // 【新增】找不到卡车，重试
                        state.retry_count++;
                        if (state.retry_count < state.max_retry_count) {
                            state.stable_frames_remaining = 1;
                        } else {
                            state.retry_count = 0;
                            state.last_dumping_bucket_box = cv::Rect(0,0,0,0);
                        }
                    }
                } else {
                    // 【新增】找不到铲斗，重试
                    state.retry_count++;
                    if (state.retry_count < state.max_retry_count) {
                        state.stable_frames_remaining = 1;
                    } else {
                        state.retry_count = 0;
                        state.last_dumping_bucket_box = cv::Rect(0,0,0,0);
                    }
                }
            }
        }

        // === E. 真实比例断崖下跌计算 ===
        if (state.is_statting && state.stat_frames_remaining > 0) {
            state.stat_frames_remaining--;

            if (state.current_truck_box.area() > 0 && !truck_boxes.empty()) {
                float min_dist = 1e9;
                cv::Rect best_t(0,0,0,0);
                float prev_cx = state.current_truck_box.x + state.current_truck_box.width / 2.0f;

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

            if (state.stat_frames_remaining == 0) {
                state.is_statting = false;
                float sum = 0; for(float r: state.ratio_buffer) sum += r;
                float avg_ratio = state.ratio_buffer.empty() ? 0.0f : (sum / state.ratio_buffer.size());

                if (state.last_avg_ratio < 0) {
                    state.last_avg_ratio = avg_ratio;
                    if (state.pending_buckets > 0) commit_pending_buckets();
                } else {
                    if (state.last_avg_ratio == 0.0f && avg_ratio == 0.0f) {
                        if (state.pending_buckets > 0) commit_pending_buckets();
                    } else {
                        float decline = state.last_avg_ratio > 0 ? (state.last_avg_ratio - avg_ratio) / state.last_avg_ratio : 0.0f;
                        if (avg_ratio == 0.0f || decline >= config.decline_thresh) {
                            cut_truck(now);
                        } else {
                            if (state.pending_buckets > 0) commit_pending_buckets();
                        }
                    }
                    state.last_avg_ratio = avg_ratio;
                }
                state.current_truck_box = cv::Rect(0,0,0,0);
            }
        }

        // === F. 快速强行合并兜底 ===
        if (state.pending_buckets > 0 && state.frames_since_bucket_empty > 15) {
            commit_pending_buckets();
        }

        // === G. 更新 UI 实时渲染框 ===
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
    void set_pipeline_timeout(void* handle, long long timeout_ms) { if (handle) ((ExcavatorPipeline*)handle)->setTimeout(timeout_ms); }
    void clear_pipeline_events(void* handle) { if (handle) ((ExcavatorPipeline*)handle)->clear_events(); }
}
