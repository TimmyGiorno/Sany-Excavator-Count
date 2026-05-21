#include "rknn_engine.h"
#include "rknn_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

RknnEngine::RknnEngine()
    : initialized_(false)
    , ctx_(nullptr)
    , in_w_(0), in_h_(0)
    , out_n_(0), out_c_(0)
    , n_classes_(4)
    , conf_thres_(0.35f)
    , iou_thres_(0.45f)
{}

RknnEngine::~RknnEngine() {
    if (ctx_) {
        rknn_destroy(static_cast<rknn_context>(ctx_));
        ctx_ = nullptr;
    }
}

bool RknnEngine::load_model(const char* model_path) {
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        fprintf(stderr, "[RknnEngine] Cannot open model: %s\n", model_path);
        return false;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* model_data = new unsigned char[fsize];
    fread(model_data, 1, fsize, fp);
    fclose(fp);

    rknn_context raw_ctx = 0;
    int ret = rknn_init(&raw_ctx, model_data, (uint32_t)fsize, 0, nullptr);
    delete[] model_data;

    if (ret != 0) {
        fprintf(stderr, "[RknnEngine] rknn_init failed, ret=%d\n", ret);
        return false;
    }
    ctx_ = reinterpret_cast<void*>(raw_ctx);

    rknn_input_output_num io_num;
    ret = rknn_query(raw_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != 0) {
        fprintf(stderr, "[RknnEngine] rknn_query IO num failed, ret=%d\n", ret);
        return false;
    }

    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    ret = rknn_query(raw_ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (ret != 0) {
        fprintf(stderr, "[RknnEngine] rknn_query input attr failed, ret=%d\n", ret);
        return false;
    }
    in_w_ = in_attr.dims[2];
    in_h_ = in_attr.dims[1];

    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    ret = rknn_query(raw_ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));
    if (ret != 0) {
        fprintf(stderr, "[RknnEngine] rknn_query output attr failed, ret=%d\n", ret);
        return false;
    }
    out_n_ = out_attr.dims[1];
    out_c_ = out_attr.dims[2];
    n_classes_ = (int)out_c_ - 4;

    fprintf(stdout, "[RknnEngine] Model loaded. input=%dx%d output=[%d,%d] classes=%d\n",
            in_w_, in_h_, out_n_, out_c_, n_classes_);

    initialized_ = true;
    return true;
}

void RknnEngine::letterbox(const uint8_t* src, int sw, int sh,
                           uint8_t* dst, int dw, int dh,
                           float& ratio, int& pad_left, int& pad_top) {
    float r = std::min((float)dw / sw, (float)dh / sh);
    ratio = r;
    int new_w = (int)(sw * r);
    int new_h = (int)(sh * r);
    pad_left = (dw - new_w) / 2;
    pad_top  = (dh - new_h) / 2;

    /* Fill with YOLO grey (114,114,114) */
    memset(dst, 114, dw * dh * 3);

    /* Nearest-neighbour scale into destination */
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_y = y * sh / new_h;
            int src_x = x * sw / new_w;
            int si = (src_y * sw + src_x) * 3;
            int di = ((y + pad_top) * dw + (x + pad_left)) * 3;
            dst[di + 0] = src[si + 0];
            dst[di + 1] = src[si + 1];
            dst[di + 2] = src[si + 2];
        }
    }
}

std::vector<DetectBox> RknnEngine::detect(const uint8_t* rgb_data,
                                          int width, int height) {
    std::vector<DetectBox> results;
    if (!initialized_) return results;

    int img_size = in_w_ * in_h_ * 3;
    uint8_t* preproc = new uint8_t[img_size];
    float ratio;
    int pad_left, pad_top;
    letterbox(rgb_data, width, height, preproc, in_w_, in_h_, ratio, pad_left, pad_top);

    rknn_context raw_ctx = static_cast<rknn_context>(ctx_);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = (uint32_t)img_size;
    inputs[0].buf   = preproc;

    int ret = rknn_inputs_set(raw_ctx, 1, inputs);
    if (ret != 0) {
        fprintf(stderr, "[RknnEngine] rknn_inputs_set failed, ret=%d\n", ret);
        delete[] preproc;
        return results;
    }

    ret = rknn_run(raw_ctx, nullptr);
    if (ret != 0) {
        fprintf(stderr, "[RknnEngine] rknn_run failed, ret=%d\n", ret);
        delete[] preproc;
        return results;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret = rknn_outputs_get(raw_ctx, 1, outputs, nullptr);
    if (ret != 0) {
        fprintf(stderr, "[RknnEngine] rknn_outputs_get failed, ret=%d\n", ret);
        delete[] preproc;
        return results;
    }

    results = decode_output((float*)outputs[0].buf, out_n_, out_c_,
                            ratio, pad_left, pad_top, width, height);

    rknn_outputs_release(raw_ctx, 1, outputs);
    delete[] preproc;
    return results;
}

std::vector<DetectBox> RknnEngine::decode_output(const float* output,
                                                  int rows, int cols,
                                                  float ratio, int pad_left, int pad_top,
                                                  int orig_w, int orig_h) {
    std::vector<DetectBox> candidates;

    for (int i = 0; i < rows; i++) {
        const float* row = output + i * cols;
        int best_cls = 0;
        float best_conf = 0.f;
        for (int c = 4; c < cols; c++) {
            if (row[c] > best_conf) {
                best_conf = row[c];
                best_cls = c - 4;
            }
        }
        if (best_conf < conf_thres_) continue;

        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];

        cx = (cx - pad_left) / ratio;
        cy = (cy - pad_top)  / ratio;
        w  = w  / ratio;
        h  = h  / ratio;

        float x1 = std::max(0.f, cx - w * 0.5f);
        float y1 = std::max(0.f, cy - h * 0.5f);
        float x2 = std::min((float)orig_w, cx + w * 0.5f);
        float y2 = std::min((float)orig_h, cy + h * 0.5f);

        if (x2 <= x1 || y2 <= y1) continue;

        candidates.push_back({x1, y1, x2, y2, best_conf, best_cls});
    }

    return nms(std::move(candidates), iou_thres_);
}

float RknnEngine::iou(const DetectBox& a, const DetectBox& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.f, ix2 - ix1);
    float ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float denom = area_a + area_b - inter;
    return (denom > 0.f) ? (inter / denom) : 0.f;
}

std::vector<DetectBox> RknnEngine::nms(std::vector<DetectBox> boxes, float thres) {
    std::vector<DetectBox> kept;
    if (boxes.empty()) return kept;

    std::sort(boxes.begin(), boxes.end(),
              [](const DetectBox& a, const DetectBox& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); j++) {
            if (suppressed[j]) continue;
            if (iou(boxes[i], boxes[j]) > thres) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}
