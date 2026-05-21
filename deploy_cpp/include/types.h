#ifndef DEPLOY_TYPES_H
#define DEPLOY_TYPES_H

#include <cstdint>

/* Shared detection box -- used by all backends */
struct DetectBox {
    float x1, y1, x2, y2;
    float confidence;
    int   class_id;
};

/* Model input/output shape queried from backend after load */
struct ModelInfo {
    int input_w;
    int input_h;
    int output_rows;   /* detections per frame */
    int output_cols;   /* 4 (bbox) + n_classes */
    int n_classes;
};

#endif
