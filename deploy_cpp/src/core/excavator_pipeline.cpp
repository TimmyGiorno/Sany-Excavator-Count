#include "excavator_pipeline.h"
#include "rknn_api.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

// ================= 宏与配置区 =================
#define YOLO_INPUT_W 640
#define YOLO_INPUT_H 640
#define YOLO_REG_MAX 16
#define SIAMESE_INPUT_SIZE 224

const float CONF_THRESH = 0.3f;
const float IOU_THRESH = 0.45f;
const float SIAMESE_THRESH = 0.75f;
const int NMS_OFFSET = 4096;

const std::vector<std::string> CLASSES = {"bucket-empty", "bucket-full", "truck", "loading", "dumping"};

// ================= 数据结构 =================
struct BBox {
    int xmin, ymin, xmax, ymax;
    float score;
    int class_id;
};

struct PipelineState {
    int total_truck_count = 0;
    int total_bucket_count = 0;
    bool bucket_full = false;
    bool dumping_active = false;
    cv::Rect last_dumping_box;
    cv::Mat last_truck_img;
    cv::Mat previous_truck_img;
    std::vector<float> last_truck_emb;
};

class ExcavatorPipeline {
private:
    rknn_context rknn_yolo = 0;
    rknn_context rknn_siamese = 0;
    PipelineState state;

    cv::VideoWriter writer;
    std::string out_video_path;
    bool is_writer_initialized = false;

    // 辅助：加载 RKNN 模型文件进入内存
    static unsigned char* load_model(const char* filename, int* model_size) {
        FILE* fp = fopen(filename, "rb");
        if (!fp) return nullptr;
        fseek(fp, 0, SEEK_END);
        int len = ftell(fp);
        unsigned char* model = (unsigned char*)malloc(len);
        fseek(fp, 0, SEEK_SET);
        if (len != fread(model, 1, len, fp)) {
            free(model);
            return nullptr;
        }
        fclose(fp);
        *model_size = len;
        return model;
    }

    // YOLO Letterbox 预处理
    cv::Mat letterbox(cv::Mat& img, float& ratio, int& dw, int& dh) {
        int h = img.rows, w = img.cols;
        ratio = std::min((float)YOLO_INPUT_W / w, (float)YOLO_INPUT_H / h);
        int new_unpad_w = std::round(w * ratio);
        int new_unpad_h = std::round(h * ratio);

        dw = (YOLO_INPUT_W - new_unpad_w) / 2;
        dh = (YOLO_INPUT_H - new_unpad_h) / 2;

        cv::Mat resized;
        if (w != new_unpad_w || h != new_unpad_h) {
            cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img.clone();
        }

        cv::Mat output;
        cv::copyMakeBorder(resized, output, dh, YOLO_INPUT_H - new_unpad_h - dh,
                           dw, YOLO_INPUT_W - new_unpad_w - dw,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        return output;
    }

    // 检查水平交集
    bool check_horizontal_overlap(const cv::Rect& b1, const cv::Rect& b2) {
        return !(b1.x + b1.width < b2.x || b2.x + b2.width < b1.x);
    }

    // 计算余弦相似度
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
    ExcavatorPipeline(const char* yolo_path, const char* siamese_path, const char* out_path) {
        if (out_path) out_video_path = out_path;

        int size = 0;
        unsigned char* yolo_model = load_model(yolo_path, &size);
        rknn_init(&rknn_yolo, yolo_model, size, 0, NULL);
        free(yolo_model);

        unsigned char* siamese_model = load_model(siamese_path, &size);
        rknn_init(&rknn_siamese, siamese_model, size, 0, NULL);
        free(siamese_model);
    }

    ~ExcavatorPipeline() {
        if (rknn_yolo) rknn_destroy(rknn_yolo);
        if (rknn_siamese) rknn_destroy(rknn_siamese);
    }

    void process(cv::Mat& frame) {
        // ========== 1. YOLO 推理 ==========
        float ratio; int dw, dh;
        cv::Mat prep_img = letterbox(frame, ratio, dw, dh);
        cv::cvtColor(prep_img, prep_img, cv::COLOR_BGR2RGB);

        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = YOLO_INPUT_W * YOLO_INPUT_H * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = prep_img.data;
        rknn_inputs_set(rknn_yolo, 1, inputs);

        rknn_run(rknn_yolo, NULL);

        // ========== 2. 完全复现 Python 的 DFL 解码与后处理 ==========
        rknn_output yolo_outputs[3];
        memset(yolo_outputs, 0, sizeof(yolo_outputs));
        for (int i = 0; i < 3; ++i) yolo_outputs[i].want_float = 1;
        rknn_outputs_get(rknn_yolo, 3, yolo_outputs, NULL);

        std::vector<BBox> results;
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> nms_scores;
        std::vector<int> nms_class_ids;

        int strides[3] = {8, 16, 32};
        int reg_max = YOLO_REG_MAX;
        int num_classes = CLASSES.size();

        for (int i = 0; i < 3; ++i) {
            int grid_w = YOLO_INPUT_W / strides[i];
            int grid_h = YOLO_INPUT_H / strides[i];
            float* out_ptr = (float*)yolo_outputs[i].buf;

            // ONNX 导出并在 RKNN 推理后，通常维度是 CHW (Channels, Height, Width)
            int map_size = grid_h * grid_w;

            for (int h = 0; h < grid_h; ++h) {
                for (int w = 0; w < grid_w; ++w) {
                    int spatial_idx = h * grid_w + w;

                    // 1. 提取得分并 Sigmoid
                    float max_score = -1.0f;
                    int best_class = -1;
                    for (int c = 0; c < num_classes; ++c) {
                        int c_idx = (4 * reg_max + c) * map_size + spatial_idx;
                        float raw_val = out_ptr[c_idx];
                        raw_val = std::max(-88.0f, std::min(88.0f, raw_val)); // 截断防止溢出
                        float score = 1.0f / (1.0f + std::exp(-raw_val));
                        if (score > max_score) {
                            max_score = score;
                            best_class = c;
                        }
                    }

                    // 2. 置信度过滤与 DFL 坐标还原
                    if (max_score > CONF_THRESH) {
                        float dfl_preds[4];
                        for (int k = 0; k < 4; ++k) {
                            float max_reg = -1e9f;
                            std::vector<float> reg_raw(reg_max);
                            for (int r = 0; r < reg_max; ++r) {
                                int r_idx = (k * reg_max + r) * map_size + spatial_idx;
                                float val = out_ptr[r_idx];
                                reg_raw[r] = val;
                                if (val > max_reg) max_reg = val;
                            }

                            // Softmax
                            float sum_exp = 0.0f;
                            float dfl_val = 0.0f;
                            for (int r = 0; r < reg_max; ++r) {
                                float exp_val = std::exp(reg_raw[r] - max_reg);
                                reg_raw[r] = exp_val;
                                sum_exp += exp_val;
                            }
                            // 点积加权
                            for (int r = 0; r < reg_max; ++r) {
                                dfl_val += (reg_raw[r] / sum_exp) * r;
                            }
                            dfl_preds[k] = dfl_val;
                        }

                        // 3. 映射回原图坐标 (逆 Letterbox 缩放)
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

        // Class-Aware NMS
        std::vector<int> indices;
        std::vector<cv::Rect> offset_boxes;
        for (size_t i = 0; i < nms_boxes.size(); ++i) {
            offset_boxes.push_back(cv::Rect(
                nms_boxes[i].x + nms_class_ids[i] * NMS_OFFSET,
                nms_boxes[i].y + nms_class_ids[i] * NMS_OFFSET,
                nms_boxes[i].width, nms_boxes[i].height
            ));
        }

        cv::dnn::NMSBoxes(offset_boxes, nms_scores, CONF_THRESH, IOU_THRESH, indices);

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

        // ========== 3. 业务逻辑状态机 (铲斗逻辑 & Siamese 卡车重识别) ==========
        std::vector<BBox> truck_boxes, bucket_boxes, dumping_boxes;
        for (const auto& r : results) {
            if (r.class_id == 2) truck_boxes.push_back(r);
            else if (r.class_id == 0 || r.class_id == 1) bucket_boxes.push_back(r);
            else if (r.class_id == 4) dumping_boxes.push_back(r);
        }

        // A. 铲斗（Bucket）状态更新
        if (!bucket_boxes.empty()) {
            auto best_bucket = bucket_boxes[0];
            for (const auto& b : bucket_boxes) if (b.score > best_bucket.score) best_bucket = b;

            cv::Rect b_rect(best_bucket.xmin, best_bucket.ymin, best_bucket.xmax - best_bucket.xmin, best_bucket.ymax - best_bucket.ymin);

            if (best_bucket.class_id == 1) { // bucket-full
                if (!state.bucket_full) {
                    bool overlap = false;
                    for (const auto& t : truck_boxes) {
                        if (check_horizontal_overlap(b_rect, cv::Rect(t.xmin, t.ymin, t.xmax - t.xmin, t.ymax - t.ymin))) overlap = true;
                    }
                    if (!overlap) state.bucket_full = true;
                }
            } else if (best_bucket.class_id == 0) { // bucket-empty
                if (state.bucket_full) {
                    bool overlap = false;
                    for (const auto& t : truck_boxes) {
                        if (check_horizontal_overlap(b_rect, cv::Rect(t.xmin, t.ymin, t.xmax - t.xmin, t.ymax - t.ymin))) overlap = true;
                    }
                    state.bucket_full = false;
                    if (overlap) state.total_bucket_count++;
                }
            }
        }

        // B. 卡车卸料与 Siamese 特征重识别
        bool has_dumping = !dumping_boxes.empty();
        if (has_dumping && state.bucket_full) {
            if (!state.dumping_active) state.dumping_active = true;
            state.last_dumping_box = cv::Rect(dumping_boxes[0].xmin, dumping_boxes[0].ymin,
                                              dumping_boxes[0].xmax - dumping_boxes[0].xmin, dumping_boxes[0].ymax - dumping_boxes[0].ymin);
        }

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
                        cv::resize(truck_crop, siamese_in, cv::Size(SIAMESE_INPUT_SIZE, SIAMESE_INPUT_SIZE));
                        cv::cvtColor(siamese_in, siamese_in, cv::COLOR_BGR2RGB);

                        rknn_input s_inputs[1];
                        memset(s_inputs, 0, sizeof(s_inputs));
                        s_inputs[0].index = 0;
                        s_inputs[0].type = RKNN_TENSOR_UINT8;
                        s_inputs[0].size = SIAMESE_INPUT_SIZE * SIAMESE_INPUT_SIZE * 3;
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

                        if (!state.last_truck_emb.empty()) {
                            float sim = cosine_similarity(state.last_truck_emb, current_emb);
                            if (sim < SIAMESE_THRESH) state.total_truck_count++;
                        } else {
                            state.total_truck_count++;
                        }

                        state.previous_truck_img = state.last_truck_img.clone();
                        state.last_truck_img = truck_crop.clone();
                        state.last_truck_emb = current_emb;

                        rknn_outputs_release(rknn_siamese, 1, s_outputs);
                    }
                }
            }
            state.dumping_active = false;
        }

        // ========== 4. 画面渲染 (图像序列回落) ==========
        if (!out_video_path.empty()) {
            int orig_h = frame.rows;
            int orig_w = frame.cols;

            if (!is_writer_initialized) {
                int display_width = orig_h / 2;
                int canvas_w = orig_w + display_width + 20;
                // 使用 0，彻底回落到存图像序列，避开 Android 阉割 FFmpeg 的坑
                int fourcc = 0;
                writer.open(out_video_path, fourcc, 25.0, cv::Size(canvas_w, orig_h));
                is_writer_initialized = true;
            }

            for (const auto& res : results) {
                cv::Scalar color(255, 255, 255);
                if (res.class_id == 0) color = cv::Scalar(255, 0, 0);
                else if (res.class_id == 1) color = cv::Scalar(0, 0, 255);
                else if (res.class_id == 2) color = cv::Scalar(0, 255, 255);
                else if (res.class_id == 3) color = cv::Scalar(0, 255, 0);
                else if (res.class_id == 4) color = cv::Scalar(255, 0, 255);

                cv::rectangle(frame, cv::Point(res.xmin, res.ymin), cv::Point(res.xmax, res.ymax), color, 2);

                std::string text = CLASSES[res.class_id] + ": " + cv::format("%.2f", res.score);
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(frame, cv::Point(res.xmin, res.ymin - text_size.height - 5),
                              cv::Point(res.xmin + text_size.width, res.ymin), color, cv::FILLED);
                cv::putText(frame, text, cv::Point(res.xmin, res.ymin - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }

            cv::rectangle(frame, cv::Point(20, 20), cv::Point(320, 110), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, "Trucks:  " + std::to_string(state.total_truck_count),
                        cv::Point(35, 55), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Buckets: " + std::to_string(state.total_bucket_count),
                        cv::Point(35, 95), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);

            if (writer.isOpened()) {
                int display_width = orig_h / 2;
                int canvas_w = orig_w + display_width + 20;
                cv::Mat output_canvas = cv::Mat::zeros(orig_h, canvas_w, CV_8UC3);

                frame.copyTo(output_canvas(cv::Rect(0, 0, orig_w, orig_h)));

                int right_x = orig_w + 10;
                int block_h = orig_h / 2 - 15;

                if (!state.previous_truck_img.empty()) {
                    cv::Mat p_resize;
                    cv::resize(state.previous_truck_img, p_resize, cv::Size(display_width, block_h));
                    p_resize.copyTo(output_canvas(cv::Rect(right_x, 10, display_width, block_h)));
                    cv::putText(output_canvas, "Previous Target", cv::Point(right_x + 10, 30),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                }

                int y_offset = 10 + block_h + 10;
                if (!state.last_truck_img.empty()) {
                    cv::Mat c_resize;
                    cv::resize(state.last_truck_img, c_resize, cv::Size(display_width, block_h));
                    c_resize.copyTo(output_canvas(cv::Rect(right_x, y_offset, display_width, block_h)));
                    cv::putText(output_canvas, "Current Target", cv::Point(right_x + 10, y_offset + 20),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
                }

                writer.write(output_canvas);
            }
        }
    }
};

// ================= C 风格导出实现 =================
void* init_pipeline(const char* yolo_path, const char* siamese_path, const char* out_video_path) {
    auto* pipeline = new ExcavatorPipeline(yolo_path, siamese_path, out_video_path);
    return (void*)pipeline;
}

void process_frame(void* handle, unsigned char* img_data, int width, int height, int channels) {
    if (!handle || !img_data) return;
    ExcavatorPipeline* pipeline = (ExcavatorPipeline*)handle;
    
    int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    cv::Mat frame(height, width, type, img_data);
    pipeline->process(frame);
}

void release_pipeline(void* handle) {
    if (handle) {
        ExcavatorPipeline* pipeline = (ExcavatorPipeline*)handle;
        delete pipeline;
    }
}