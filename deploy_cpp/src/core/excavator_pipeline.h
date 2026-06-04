#ifndef EXCAVATOR_PIPELINE_H
#define EXCAVATOR_PIPELINE_H

// ================= 跨平台动态库导出宏定义 =================
#if defined(_WIN32) || defined(_WIN64)
    #ifdef EXCAVATOR_EXPORTS
        #define EXCAVATOR_API __declspec(dllexport)
    #else
        #define EXCAVATOR_API __declspec(dllimport)
    #endif
#else
    #define EXCAVATOR_API __attribute__((visibility("default")))
#endif
// ==========================================================

#ifdef __cplusplus
extern "C" {
#endif

    // 1. 初始化流水线
    EXCAVATOR_API void* init_pipeline_from_memory(
        const void* yolo_data, int yolo_size, const void* siamese_data, int siamese_size);

    // 2. 推理单帧图像
    EXCAVATOR_API void process_frame(void* handle, unsigned char* img_data, int width, int height, int channels);

    // 3. 释放资源
    EXCAVATOR_API void release_pipeline(void* handle);

#ifdef __cplusplus
}
#endif

#endif // EXCAVATOR_PIPELINE_H