#ifndef RKNN_API_H
#define RKNN_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define RKNN_MAX_IO_NUM  16

// Tensor format
typedef enum {
    RKNN_TENSOR_NCHW  = 0,
    RKNN_TENSOR_NHWC  = 1,
    RKNN_TENSOR_NC1HWC2 = 2,
} rknn_tensor_format;

// Tensor data type
typedef enum {
    RKNN_TENSOR_FLOAT32 = 0,
    RKNN_TENSOR_FLOAT16 = 1,
    RKNN_TENSOR_INT8    = 2,
    RKNN_TENSOR_UINT8   = 3,
    RKNN_TENSOR_INT16   = 4,
    RKNN_TENSOR_UINT16  = 5,
    RKNN_TENSOR_INT32   = 6,
    RKNN_TENSOR_UINT32  = 7,
    RKNN_TENSOR_INT64   = 8,
    RKNN_TENSOR_BOOL    = 9,
} rknn_tensor_type;

typedef struct {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[4];
    char     name[256];
    uint32_t n_elems;
    uint32_t size;
    uint32_t fmt;
    uint32_t type;
    uint32_t qnt_type;
    int8_t   fl;
    int8_t   zp;
    float    scale;
    uint32_t w_stride;
    uint32_t size_with_stride;
    uint32_t pass_through;
    uint32_t h_stride;
} rknn_tensor_attr;

typedef struct {
    uint32_t index;
    void    *buf;
    uint32_t size;
    uint32_t pass_through;
    uint32_t type;
    uint32_t fmt;
} rknn_input;

typedef struct {
    uint8_t   want_float;
    uint8_t   is_prealloc;
    uint32_t  index;
    void     *buf;
    uint32_t  size;
} rknn_output;

typedef struct {
    char api_version[32];
    char drv_version[32];
} rknn_sdk_version;

typedef struct {
    uint32_t n_input;
    uint32_t n_output;
} rknn_input_output_num;

typedef void *rknn_context;

int rknn_init(rknn_context* ctx, void* model, uint32_t size, uint32_t flag, void* opt);
int rknn_destroy(rknn_context ctx);
int rknn_query(rknn_context ctx, int query_cmd, void* info, uint32_t size);
int rknn_inputs_set(rknn_context ctx, uint32_t n_inputs, rknn_input inputs[]);
int rknn_run(rknn_context ctx, void* extend);
int rknn_outputs_get(rknn_context ctx, uint32_t n_outputs, rknn_output outputs[], void* extend);
int rknn_outputs_release(rknn_context ctx, uint32_t n_ouputs, rknn_output outputs[]);

#define RKNN_QUERY_IN_OUT_NUM     0
#define RKNN_QUERY_INPUT_ATTR     1
#define RKNN_QUERY_OUTPUT_ATTR    2
#define RKNN_QUERY_PERF_DETAIL    3
#define RKNN_QUERY_PERF_RUN       4
#define RKNN_QUERY_SDK_VERSION    5
#define RKNN_QUERY_MEM_SIZE       6
#define RKNN_QUERY_CUSTOM_STRING  7

#ifdef __cplusplus
}
#endif

#endif
