#include "rknn_api.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>

// ================= 配置与数据结构 =================
// 将原有的宏定义改为可通过 JNI 动态配置的结构体
struct PipelineConfig {
    int yolo_input_w = 640;
    int yolo_input_h = 640;
    int yolo_reg_max = 16;
    int siamese_input_size = 224;

    float conf_thresh = 0.3f;
    float iou_thresh = 0.45f;
    float siamese_thresh = 0.75f;
    int nms_offset = 4096;
};

const std::vector<std::string> CLASSES = {"bucket-empty", "bucket-full", "truck", "loading", "dumping"};

struct BBox {
    int xmin, ymin, xmax, ymax;
    float score;
    int class_id;
};

struct PipelineState {
    std::string ticket_id = "";        // 当前装车票号 (时间戳生成)
    int total_truck_count = 0;         // 绝对总车数
    int total_bucket_count = 0;        // 绝对总铲数 (永不重置)

    bool bucket_full = false;
    bool dumping_active = false;
    cv::Rect last_dumping_box;

    // 底图滚动更新，防止特征漂移
    cv::Mat reference_truck_img;
    std::vector<float> reference_truck_emb;

    // 暴露给 Java 层的当前帧状态 (用于 UI 绘制和业务判断)
    cv::Rect current_bucket_box = cv::Rect(0,0,0,0);
    int current_bucket_type = -1;  // -1无, 0空, 1满
    cv::Rect current_truck_box = cv::Rect(0,0,0,0);

    // 关键事件触发器：当此标志为 true 时，Java 层应记录新车的起始斗数
    bool is_new_truck_entered = false;
};

class ExcavatorPipeline {
private:
    rknn_context rknn_yolo = 0;
    rknn_context rknn_siamese = 0;
    PipelineState state;
    PipelineConfig config;

    // 生成时间戳票号
    std::string generate_ticket_id() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        return "TKT_" + std::to_string(ms);
    }

    // YOLO Letterbox 预处理
    cv::Mat letterbox(cv::Mat& img, float& ratio, int& dw, int& dh) {
        int h = img.rows, w = img.cols;
        ratio = std::min((float)config.yolo_input_w / w, (float)config.yolo_input_h / h);
        int new_unpad_w = std::round(w * ratio);
        int new_unpad_h = std::round(h * ratio);

        dw = (config.yolo_input_w - new_unpad_w) / 2;
        dh = (config.yolo_input_h - new_unpad_h) / 2;

        cv::Mat resized;
        if (w != new_unpad_w || h != new_unpad_h) {
            cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img.clone();
        }

        cv::Mat output;
        cv::copyMakeBorder(resized, output, dh, config.yolo_input_h - new_unpad_h - dh,
                           dw, config.yolo_input_w - new_unpad_w - dw,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return output;
    }

    bool check_horizontal_overlap(const cv::Rect& b1, const cv::Rect& b2) {
        return !(b1.x + b1.width < b2.x || b2.x + b2.width < b1.x);
    }

    float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.size() != v2.size() || v1.empty()) return 0.0f;
        float dot = 0.0f, denom1 = 0.0f, denom2 = 0.0f;
        for (size_t i = 0; i < v1.size(); ++i) {
            dot += v1[i] * v2[i];
            denom1 += v1[i] * v1[i];
            denom2 += v2[i] * v2[i];
        }
        return dot / (std::sqrt(denom1) * std::sqrt(denom2));
    }

public:
    ExcavatorPipeline(const void* yolo_data, const int yolo_size, const void* siamese_data, const int siamese_size) {
        if (yolo_data && yolo_size > 0) {
            rknn_init(&rknn_yolo, const_cast<void *>(yolo_data), yolo_size, 0, nullptr);
        }
        if (siamese_data && siamese_size > 0) {
            rknn_init(&rknn_siamese, const_cast<void *>(siamese_data), siamese_size, 0, nullptr);
        }
    }

    ~ExcavatorPipeline() {
        if (rknn_yolo) rknn_destroy(rknn_yolo);
        if (rknn_siamese) rknn_destroy(rknn_siamese);
    }

    void updateConfig(const PipelineConfig& new_config) {
        this->config = new_config;
    }

    void restoreState(const std::string& ticket_id, int bucket_count, const float* feature_data, int feature_dim) {
        state.ticket_id = ticket_id;
        state.total_bucket_count = bucket_count;

        // 恢复特征向量
        if (feature_data && feature_dim > 0) {
            state.reference_truck_emb.assign(feature_data, feature_data + feature_dim);
        }

        // 重置临时状态，防止因为上次断电时的残留状态导致死锁
        state.dumping_active = false;
        state.bucket_full = false;
    }

    const PipelineState& getState() const {
        return state;
    }

    void process(cv::Mat& frame) {
        // 每次推理前，重置事件触发器，避免向 Java 层发送重复事件
        state.current_bucket_type = -1;
        state.current_bucket_box = cv::Rect(0,0,0,0);
        state.current_truck_box = cv::Rect(0,0,0,0);
        state.is_new_truck_entered = false;

        // ========== 1. YOLO 推理 ==========
        float ratio; int dw, dh;
        cv::Mat prep_img = letterbox(frame, ratio, dw, dh);
        cv::cvtColor(prep_img, prep_img, cv::COLOR_BGR2RGB);

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = config.yolo_input_w * config.yolo_input_h * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = prep_img.data;
        rknn_inputs_set(rknn_yolo, 1, inputs);

        rknn_run(rknn_yolo, NULL);

        // ========== 2. DFL 解码与后处理 ==========
        rknn_output yolo_outputs[3];
        memset(yolo_outputs, 0, sizeof(yolo_outputs));
        for (int i = 0; i < 3; ++i) yolo_outputs[i].want_float = 1;
        rknn_outputs_get(rknn_yolo, 3, yolo_outputs, NULL);

        std::vector<BBox> results;
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
                        if (score > max_score) {
                            max_score = score;
                            best_class = c;
                        }
                    }

                    if (max_score > config.conf_thresh) {
                        float dfl_preds[4];
                        for (int k = 0; k < 4; ++k) {
                            float max_reg = -1e9f;
                            std::vector<float> reg_raw(config.yolo_reg_max);
                            for (int r = 0; r < config.yolo_reg_max; ++r) {
                                int r_idx = (k * config.yolo_reg_max + r) * map_size + spatial_idx;
                                float val = out_ptr[r_idx];
                                reg_raw[r] = val;
                                if (val > max_reg) max_reg = val;
                            }

                            float sum_exp = 0.0f;
                            float dfl_val = 0.0f;
                            for (int r = 0; r < config.yolo_reg_max; ++r) {
                                float exp_val = std::exp(reg_raw[r] - max_reg);
                                reg_raw[r] = exp_val;
                                sum_exp += exp_val;
                            }
                            for (int r = 0; r < config.yolo_reg_max; ++r) {
                                dfl_val += (reg_raw[r] / sum_exp) * r;
                            }
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
            offset_boxes.push_back(cv::Rect(
                nms_boxes[i].x + nms_class_ids[i] * config.nms_offset,
                nms_boxes[i].y + nms_class_ids[i] * config.nms_offset,
                nms_boxes[i].width, nms_boxes[i].height
            ));
        }

        cv::dnn::NMSBoxes(offset_boxes, nms_scores, config.conf_thresh, config.iou_thresh, indices);

        for (int idx : indices) {
            BBox box;
            box.xmin = nms_boxes[idx].x;
            box.ymin = nms_boxes[idx].y;
            box.xmax = nms_boxes[idx].x + nms_boxes[idx].width;
            box.ymax = nms_boxes[idx].y + nms_boxes[idx].height;
            box.score = nms_scores[idx];
            box.class_id = nms_class_ids[idx];
            results.push_back(box);
        }

        // ========== 3. 业务逻辑状态机 ==========
        std::vector<BBox> truck_boxes, bucket_boxes, dumping_boxes;
        for (const auto& r : results) {
            if (r.class_id == 2) truck_boxes.push_back(r);
            else if (r.class_id == 0 || r.class_id == 1) bucket_boxes.push_back(r);
            else if (r.class_id == 4) dumping_boxes.push_back(r);
        }

        if (!truck_boxes.empty()) {
            auto best_truck = truck_boxes[0];
            state.current_truck_box = cv::Rect(best_truck.xmin, best_truck.ymin, best_truck.xmax - best_truck.xmin, best_truck.ymax - best_truck.ymin);
        }

        // A. 铲斗状态更新 (保持全局独立累加，不做清零处理)
        if (!bucket_boxes.empty()) {
            auto best_bucket = bucket_boxes[0];
            for (const auto& b : bucket_boxes) if (b.score > best_bucket.score) best_bucket = b;

            cv::Rect b_rect(best_bucket.xmin, best_bucket.ymin, best_bucket.xmax - best_bucket.xmin, best_bucket.ymax - best_bucket.ymin);
            state.current_bucket_box = b_rect;
            state.current_bucket_type = best_bucket.class_id;

            if (best_bucket.class_id == 1) {
                if (!state.bucket_full) {
                    bool overlap = false;
                    for (const auto& t : truck_boxes) {
                        if (check_horizontal_overlap(b_rect, cv::Rect(t.xmin, t.ymin, t.xmax - t.xmin, t.ymax - t.ymin))) overlap = true;
                    }
                    if (!overlap) state.bucket_full = true;
                }
            } else if (best_bucket.class_id == 0) {
                if (state.bucket_full) {
                    bool overlap = false;
                    for (const auto& t : truck_boxes) {
                        if (check_horizontal_overlap(b_rect, cv::Rect(t.xmin, t.ymin, t.xmax - t.xmin, t.ymax - t.ymin))) overlap = true;
                    }
                    state.bucket_full = false;
                    if (overlap) state.total_bucket_count++; // 全局计数，交由 Java 层去做减法
                }
            }
        }

        // B. 卡车卸料与孪生网络特征重识别 (原版完美 FSM)
        bool has_dumping = !dumping_boxes.empty();

        if (has_dumping && !state.bucket_full) return; // 防抖，未满斗不判定

        if (has_dumping && state.bucket_full) {
            if (!state.dumping_active) state.dumping_active = true;
            state.last_dumping_box = cv::Rect(dumping_boxes[0].xmin, dumping_boxes[0].ymin,
                                              dumping_boxes[0].xmax - dumping_boxes[0].xmin, dumping_boxes[0].ymax - dumping_boxes[0].ymin);
        }

        // 倒土彻底结束（尘埃落定）且土已入车，触发截图比对
        if (!has_dumping && state.dumping_active && !state.bucket_full) {
            if (state.last_dumping_box.area() > 0 && !truck_boxes.empty()) {
                float dump_cx = state.last_dumping_box.x + state.last_dumping_box.width / 2.0f;
                BBox closest_truck; float min_dist = 1e9;
                bool found = false;

                for (const auto& t : truck_boxes) {
                    cv::Rect t_rect(t.xmin, t.ymin, t.xmax - t.xmin, t.ymax - t.ymin);
                    if (check_horizontal_overlap(state.last_dumping_box, t_rect)) {
                        float dist = std::abs((t.xmin + (t.xmax - t.xmin) / 2.0f) - dump_cx);
                        if (dist < min_dist) { min_dist = dist; closest_truck = t; found = true; }
                    }
                }

                if (found) {
                    int w = closest_truck.xmax - closest_truck.xmin;
                    int h = closest_truck.ymax - closest_truck.ymin;
                    int size = std::max(w, h);
                    int cx = closest_truck.xmin + w / 2;
                    int cy = closest_truck.ymin + h / 2;

                    cv::Rect crop_rect(cx - size/2, cy - size/2, size, size);
                    crop_rect &= cv::Rect(0, 0, frame.cols, frame.rows);

                    cv::Mat truck_crop = frame(crop_rect);
                    if (!truck_crop.empty()) {
                        cv::Mat siamese_in;
                        cv::resize(truck_crop, siamese_in, cv::Size(config.siamese_input_size, config.siamese_input_size));
                        cv::cvtColor(siamese_in, siamese_in, cv::COLOR_BGR2RGB);

                        rknn_input s_inputs[1];
                        memset(s_inputs, 0, sizeof(s_inputs));
                        s_inputs[0].index = 0;
                        s_inputs[0].type = RKNN_TENSOR_UINT8;
                        s_inputs[0].size = config.siamese_input_size * config.siamese_input_size * 3;
                        s_inputs[0].fmt = RKNN_TENSOR_NHWC;
                        s_inputs[0].buf = siamese_in.data;
                        rknn_inputs_set(rknn_siamese, 1, s_inputs);

                        rknn_run(rknn_siamese, NULL);

                        rknn_output s_outputs[1];
                        memset(s_outputs, 0, sizeof(s_outputs));
                        s_outputs[0].want_float = 1;
                        rknn_outputs_get(rknn_siamese, 1, s_outputs, NULL);

                        float* emb_data = (float*)s_outputs[0].buf;
                        std::vector<float> current_emb(emb_data, emb_data + 256);

                        if (!state.reference_truck_emb.empty()) {
                            float sim = cosine_similarity(state.reference_truck_emb, current_emb);
                            if (sim < config.siamese_thresh) {
                                // 更换新车
                                state.total_truck_count++;
                                state.ticket_id = generate_ticket_id();
                                state.is_new_truck_entered = true;
                            }
                        } else {
                            // 系统启动后的第一辆车
                            state.total_truck_count++;
                            state.ticket_id = generate_ticket_id();
                            state.is_new_truck_entered = true;
                        }

                        // 无论换车与否，滚动更新参考特征以应对装载造成的特征漂移
                        state.reference_truck_img = truck_crop.clone();
                        state.reference_truck_emb = current_emb;

                        rknn_outputs_release(rknn_siamese, 1, s_outputs);
                    }
                }
            }
            state.dumping_active = false;
        }
    }
};

// ================= C 风格导出实现 (适配 JNI) =================
extern "C" {
    void* init_pipeline_from_memory(const void* yolo_data, const int yolo_size, const void* siamese_data, const int siamese_size) {
        return new ExcavatorPipeline(yolo_data, yolo_size, siamese_data, siamese_size);
    }

    void process_frame(void* handle, unsigned char* img_data, int width, int height, int channels) {
        if (!handle || !img_data) return;
        int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
        cv::Mat frame(height, width, type, img_data);
        ((ExcavatorPipeline*)handle)->process(frame);
    }

    void* get_pipeline_state(void* handle) {
        if (!handle) return nullptr;
        return (void*)&(((ExcavatorPipeline*)handle)->getState());
    }

    // 新增：允许 Java 运行时动态调整参数
    void update_pipeline_config(void* handle, float conf_thresh, float iou_thresh, float siamese_thresh) {
        if (!handle) return;
        PipelineConfig cfg;
        cfg.conf_thresh = conf_thresh;
        cfg.iou_thresh = iou_thresh;
        cfg.siamese_thresh = siamese_thresh;
        ((ExcavatorPipeline*)handle)->updateConfig(cfg);
    }

    void release_pipeline(void* handle) {
        if (handle) delete (ExcavatorPipeline*)handle;
    }

    void restore_pipeline_state(void* handle, const char* ticket_id, int bucket_count, const float* feature_data, int feature_dim) {
        if (handle) {
            std::string t_id = ticket_id ? std::string(ticket_id) : "";
            ((ExcavatorPipeline*)handle)->restoreState(t_id, bucket_count, feature_data, feature_dim);
        }
    }
}