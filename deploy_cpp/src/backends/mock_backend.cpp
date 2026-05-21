#ifdef USE_MOCK

#include "../../include/inference_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

/*
   Mock backend: returns synthetic detection boxes for pipeline verification.
   No NPU / ONNX dependency -- exercises only the C++ pre/post/NMS path.
*/

class MockBackend : public IBackend {
public:
    bool load(const char* model_path, ModelInfo& info) override {
        info.input_w    = 480;
        info.input_h    = 480;
        info.output_rows = 8400;
        info.output_cols = 8;   /* 4 bbox + 4 class scores */
        info.n_classes  = 4;
        fprintf(stdout, "[MockBackend] Loaded. in=%dx%d out=[%d,%d]\n",
                info.input_w, info.input_h, info.output_rows, info.output_cols);
        return true;
    }

    bool infer(const uint8_t* input, int input_size,
               float*& output, int& rows, int& cols) override {
        rows = 8400;
        cols = 8;
        size_t bytes = rows * cols * sizeof(float);
        output = (float*)malloc(bytes);
        memset(output, 0, bytes);

        /*
           Inject 3 synthetic detections at positions [0], [1], [2].
           YOLO output row format: [cx, cy, w, h, cls0_conf, cls1_conf, cls2_conf, cls3_conf]
        */
        float* row = output;
        /* truck-full at (100, 80, 200, 180) in 480x480 letterbox space */
        float cx = 150.f, cy = 130.f, bw = 100.f, bh = 100.f;
        row[0] = cx;  row[1] = cy;  row[2] = bw;  row[3] = bh;
        row[4] = 0.f;  row[5] = 0.f;  row[6] = 0.f;  row[7] = 0.9f;  /* class 3 */

        /* bucket-full at (300, 250, 350, 290) */
        row = output + cols;
        cx = 325.f; cy = 270.f; bw = 50.f; bh = 40.f;
        row[0] = cx;  row[1] = cy;  row[2] = bw;  row[3] = bh;
        row[4] = 0.f;  row[5] = 0.9f;  row[6] = 0.f;  row[7] = 0.f;  /* class 1 */

        /* bucket-empty at (310, 260, 340, 280) -- overlaps with above, NMS should kill it */
        row = output + 2 * cols;
        cx = 325.f; cy = 270.f; bw = 30.f; bh = 20.f;
        row[0] = cx;  row[1] = cy;  row[2] = bw;  row[3] = bh;
        row[4] = 0.8f;  row[5] = 0.f;  row[6] = 0.f;  row[7] = 0.f;  /* class 0, lower conf */

        return true;
    }

    void release_output(float* output) override { free(output); }
    void unload() override {}
};

IBackend* create_mock_backend() { return new MockBackend(); }

#endif /* USE_MOCK */
