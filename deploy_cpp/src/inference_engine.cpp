#include "../include/inference_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>

InferenceEngine::InferenceEngine()
    : initialized_(false)
    , backend_(nullptr)
    , info_()
    , conf_thres_(0.35f)
    , iou_thres_(0.45f)
{}

InferenceEngine::~InferenceEngine() {
    if (backend_) {
        backend_->unload();
        delete backend_;
        backend_ = nullptr;
    }
}

bool InferenceEngine::load_model(IBackend* backend, const char* model_path) {
    if (backend_) {
        backend_->unload();
        delete backend_;
    }
    backend_ = backend;
    if (!backend_) return false;

    if (!backend_->load(model_path, info_)) {
        fprintf(stderr, "[InferenceEngine] Backend load failed.\n");
        return false;
    }

    initialized_ = true;
    return true;
}

/* ---------- Preprocessing (backend-agnostic) ---------- */

void InferenceEngine::letterbox(const uint8_t* src, int sw, int sh,
                                uint8_t* dst, int dw, int dh,
                                float& ratio, int& pad_left, int& pad_top) {
    float r = std::min((float)dw / sw, (float)dh / sh);
    ratio = r;
    int new_w = (int)(sw * r);
    int new_h = (int)(sh * r);
    pad_left = (dw - new_w) / 2;
    pad_top  = (dh - new_h) / 2;

    memset(dst, 114, dw * dh * 3);

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

/* ---------- Full pipeline: preprocess -> infer -> decode -> NMS ---------- */

std::vector<DetectBox> InferenceEngine::detect(const uint8_t* rgb_data,
                                                int width, int height) {
    std::vector<DetectBox> results;
    if (!initialized_ || !backend_) return results;

    int img_size = info_.input_w * info_.input_h * 3;
    uint8_t* preproc = new uint8_t[img_size];
    float ratio;
    int pad_left, pad_top;
    letterbox(rgb_data, width, height, preproc,
              info_.input_w, info_.input_h, ratio, pad_left, pad_top);

    float* output = nullptr;
    int rows = 0, cols = 0;

    if (!backend_->infer(preproc, img_size, output, rows, cols)) {
        fprintf(stderr, "[InferenceEngine] Backend inference failed.\n");
        delete[] preproc;
        return results;
    }

    results = decode_output(output, rows, cols,
                            ratio, pad_left, pad_top, width, height);

    backend_->release_output(output);
    delete[] preproc;
    return results;
}

/* ---------- Post-processing (backend-agnostic) ---------- */

std::vector<DetectBox> InferenceEngine::decode_output(const float* output,
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

float InferenceEngine::iou(const DetectBox& a, const DetectBox& b) {
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

std::vector<DetectBox> InferenceEngine::nms(std::vector<DetectBox> boxes, float thres) {
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
            if (iou(boxes[i], boxes[j]) > thres) suppressed[j] = true;
        }
    }
    return kept;
}
