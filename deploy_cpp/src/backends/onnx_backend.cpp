#ifdef USE_ONNX

#include "../../include/inference_engine.h"
#include <cstdio>

/*
   ONNX Runtime backend (stub).
   To activate:
     1. Install ONNX Runtime SDK in 3rdparty/onnxruntime/<platform>/
     2. Build with -DUSE_ONNX=ON
     3. Uncomment the #include <onnxruntime_cxx_api.h> below
     4. Replace stub bodies with real ONNX Runtime API calls
*/

// #include <onnxruntime_cxx_api.h>

class OnnxBackend : public IBackend {
public:
    bool load(const char* model_path, ModelInfo& info) override {
        fprintf(stderr, "[OnnxBackend] STUB -- implement ONNX Runtime loading here.\n");
        info.input_w    = 480;
        info.input_h    = 480;
        info.output_rows = 8400;
        info.output_cols = 8;
        info.n_classes  = 4;
        return false;  /* change to true once implemented */
    }

    bool infer(const uint8_t* input, int input_size,
               float*& output, int& rows, int& cols) override {
        output = nullptr;
        rows = 0;
        cols = 0;
        return false;
    }

    void release_output(float* output) override {}
    void unload() override {}
};

IBackend* create_onnx_backend() { return new OnnxBackend(); }

#endif /* USE_ONNX */
