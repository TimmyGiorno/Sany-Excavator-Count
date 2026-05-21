#ifndef RKNN_ENGINE_H
#define RKNN_ENGINE_H

#include <string>
#include <vector>
#include <cstdint>

struct DetectBox {
    float x1, y1, x2, y2;
    float confidence;
    int   class_id;
};

class RknnEngine {
public:
    RknnEngine();
    ~RknnEngine();

    bool load_model(const char* model_path);
    bool is_ready() const { return initialized_; }

    std::vector<DetectBox> detect(const uint8_t* rgb_data,
                                  int width, int height);

    int input_width()  const { return in_w_; }
    int input_height() const { return in_h_; }

private:
    bool initialized_;
    void* ctx_;

    int in_w_, in_h_;
    int out_n_, out_c_;
    int n_classes_;

    float conf_thres_;
    float iou_thres_;

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

#endif /* RKNN_ENGINE_H */
