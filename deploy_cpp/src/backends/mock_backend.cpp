#ifdef USE_MOCK

#include "../../include/inference_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Mock backend: allocates fake output of known shape, fills with 0.0.
   Used for pipeline throughput benchmarking with zero NPU dependency. */

class MockBackend : public IBackend {
public:
    bool load(const char* model_path, ModelInfo& info) override {
        info.input_w    = 480;
        info.input_h    = 480;
        info.output_rows = 8400;
        info.output_cols = 8;   /* 4 bbox + 4 classes */
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
        return true;
    }

    void release_output(float* output) override { free(output); }
    void unload() override {}
};

IBackend* create_mock_backend() { return new MockBackend(); }

#endif /* USE_MOCK */
