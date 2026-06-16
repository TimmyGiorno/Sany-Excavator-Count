#pragma once

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
    EXCAVATOR_API void* init_pipeline_from_memory(const void* yolo_data, int yolo_size);

    // 2. 推理单帧图像
    EXCAVATOR_API void process_frame(void* handle, unsigned char* img_data, int width, int height, int channels);

    // 3. 读取当前状态
    EXCAVATOR_API void* get_pipeline_state(void* handle);

    // 4. 释放资源
    EXCAVATOR_API void release_pipeline(void* handle);

    // 5. 动态调参
    EXCAVATOR_API void update_pipeline_config(void* handle, float conf_thresh, float iou_thresh);

    // 6. 断电恢复
    EXCAVATOR_API void restore_pipeline_state(void* handle, const char* ticket_id, int bucket_count, float last_mineral_ratio);

    // 7. 动态设置超时时间
    EXCAVATOR_API void set_pipeline_timeout(void* handle, long long timeout_ms);

    // 8. 清空事件队列
    EXCAVATOR_API void clear_pipeline_events(void* handle);

#ifdef __cplusplus
}
#endif