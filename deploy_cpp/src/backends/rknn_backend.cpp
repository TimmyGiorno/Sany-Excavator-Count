#ifdef USE_RKNN

#include "../../include/inference_engine.h"
#include "../../jni/rknn_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

class RknnBackend : public IBackend {
public:
    RknnBackend() : ctx_(0) {}
    ~RknnBackend() override { unload(); }

    bool load(const char* model_path, ModelInfo& info) override {
        FILE* fp = fopen(model_path, "rb");
        if (!fp) {
            fprintf(stderr, "[RknnBackend] Cannot open: %s\n", model_path);
            return false;
        }
        fseek(fp, 0, SEEK_END);
        long fsize = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        unsigned char* data = new unsigned char[fsize];
        fread(data, 1, fsize, fp);
        fclose(fp);

        rknn_context raw_ctx = 0;
        int ret = rknn_init(&raw_ctx, data, (uint32_t)fsize, 0, nullptr);
        delete[] data;

        if (ret != 0) {
            fprintf(stderr, "[RknnBackend] rknn_init failed, ret=%d\n", ret);
            return false;
        }
        ctx_ = raw_ctx;

        rknn_input_output_num io_num;
        rknn_query(raw_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

        rknn_tensor_attr in_attr = {};
        in_attr.index = 0;
        rknn_query(raw_ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));

        info.input_w = (int)in_attr.dims[2];
        info.input_h = (int)in_attr.dims[1];

        rknn_tensor_attr out_attr = {};
        out_attr.index = 0;
        rknn_query(raw_ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr));

        info.output_rows = (int)out_attr.dims[1];
        info.output_cols = (int)out_attr.dims[2];
        info.n_classes   = (int)out_attr.dims[2] - 4;
        last_rows_ = info.output_rows;
        last_cols_ = info.output_cols;

        fprintf(stdout, "[RknnBackend] Loaded. in=%dx%d out=[%d,%d] cls=%d\n",
                info.input_w, info.input_h, info.output_rows, info.output_cols, info.n_classes);
        return true;
    }

    bool infer(const uint8_t* input, int input_size,
               float*& output, int& rows, int& cols) override {
        rknn_input rk_in[1] = {};
        rk_in[0].index = 0;
        rk_in[0].type  = RKNN_TENSOR_UINT8;
        rk_in[0].fmt   = RKNN_TENSOR_NHWC;
        rk_in[0].size  = (uint32_t)input_size;
        rk_in[0].buf   = (void*)input;

        if (rknn_inputs_set(ctx_, 1, rk_in) != 0) return false;
        if (rknn_run(ctx_, nullptr) != 0) return false;

        rknn_output rk_out[1] = {};
        rk_out[0].want_float = 1;
        if (rknn_outputs_get(ctx_, 1, rk_out, nullptr) != 0) return false;

        output = (float*)rk_out[0].buf;
        rows   = last_rows_;
        cols   = last_cols_;
        return true;
    }

    void release_output(float* output) override {
        rknn_output rk_out[1] = {};
        rk_out[0].buf = output;
        rk_out[0].want_float = 1;
        rknn_outputs_release(ctx_, 1, rk_out);
    }

    void unload() override {
        if (ctx_) { rknn_destroy(ctx_); ctx_ = 0; }
    }

    /* Set output dims for the next infer call (called by engine after load) */
private:
    rknn_context ctx_;
    int last_rows_ = 0;
    int last_cols_ = 0;
};

/* Factory function -- defined here so rknn_api.h is only included in this TU */
IBackend* create_rknn_backend() { return new RknnBackend(); }

#endif /* USE_RKNN */
