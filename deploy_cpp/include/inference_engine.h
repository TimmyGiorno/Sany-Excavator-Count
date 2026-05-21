#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

/* ----------------------------------------------------------------
   Abstract backend: one implementation per hardware target
   ---------------------------------------------------------------- */
class IBackend {
public:
    virtual ~IBackend() = default;

    /* Load model file. Returns true on success and fills ModelInfo. */
    virtual bool load(const char* model_path, ModelInfo& info) = 0;

    /* Run inference on preprocessed uint8 NHWC image [H*W*3].
       On success, sets output pointer, rows, cols. Caller must call release_output(). */
    virtual bool infer(const uint8_t* input, int input_size,
                       float*& output, int& rows, int& cols) = 0;

    /* Release the output buffer obtained from the last infer() call. */
    virtual void release_output(float* output) = 0;

    /* Fully unload the model and release backend resources. */
    virtual void unload() = 0;
};

/* ----------------------------------------------------------------
   InferenceEngine: generic pre/post-processing + NMS
   ---------------------------------------------------------------- */
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    /* Takes ownership of a backend instance. */
    bool load_model(IBackend* backend, const char* model_path);

    bool is_ready() const { return initialized_; }
    int  input_width()  const { return info_.input_w; }
    int  input_height() const { return info_.input_h; }

    /* Full pipeline: preprocess -> infer -> decode -> NMS */
    std::vector<DetectBox> detect(const uint8_t* rgb_data,
                                  int width, int height);

    /* Per-class thresholds */
    void set_conf_threshold(float t) { conf_thres_ = t; }
    void set_iou_threshold(float t)  { iou_thres_ = t; }

private:
    bool        initialized_;
    IBackend*   backend_;
    ModelInfo   info_;

    float conf_thres_;
    float iou_thres_;

    /* Generic (backend-agnostic) helpers */
    void letterbox(const uint8_t* src, int sw, int sh,
                   uint8_t* dst, int dw, int dh,
                   float& ratio, int& pad_left, int& pad_top);

    std::vector<DetectBox> decode_output(const float* output,
                                         int rows, int cols,
                                         float ratio, int pad_left, int pad_top,
                                         int orig_w, int orig_h);

    static float iou(const DetectBox& a, const DetectBox& b);
    std::vector<DetectBox> nms(std::vector<DetectBox> boxes, float thres);
};

#endif
