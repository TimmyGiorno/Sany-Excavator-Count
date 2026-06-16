#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include "excavator_pipeline.h"

// ================= 服务器事件结构 (与 pipeline 保持 100% 相同) =================
struct BucketEvent {
    std::string ticket_id;
    int total_truck_count;
    int current_bucket_count;
    long long dump_start_time;
    long long dump_end_time;
};

struct TruckEvent {
    std::string ticket_id;
    int total_truck_count;
    int total_bucket_count;
    long long load_start_time;
    long long load_end_time;
};

struct TimeoutEvent {
    std::string ticket_id;
};

struct PendingBucket {
    long long dump_start_time;
    long long dump_end_time;
};

// ================= 状态机与缓存区 (与 pipeline 保持 100% 内存对齐) =================
struct PipelineState {
    std::string ticket_id;
    int total_truck_count;
    int total_bucket_count;
    int current_truck_buckets;

    int pending_buckets;
    int frames_since_bucket_empty;
    std::vector<PendingBucket> pending_queue;
    
    bool has_pushed_timeout;

    bool bucket_full;
    bool dumping_active;
    int dumping_frame_count;

    long long current_dump_start_time;
    long long truck_load_start_time;
    long long truck_load_end_time;
    long long last_dump_end_time;
    long long last_action_time;

    bool is_truck_active;

    cv::Rect last_dumping_box;
    cv::Rect current_truck_box;
    cv::Rect last_dumping_bucket_box;

    int stable_frames_remaining;
    bool is_statting;
    int stat_frames_remaining;
    std::vector<float> ratio_buffer;
    float last_avg_ratio;

    std::vector<BucketEvent> pending_bucket_events;
    std::vector<TruckEvent> pending_truck_events;
    std::vector<TimeoutEvent> pending_timeout_events;
};

std::vector<unsigned char> load_file_to_memory(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        std::cerr << "❌ 无法读取文件: " << path << std::endl;
        return {};
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<unsigned char> buffer(size);
    if (fread(buffer.data(), 1, size, fp) != size) {
        fclose(fp); return {};
    }
    fclose(fp);
    return buffer;
}

// ========================================================
// 辅助函数：消费并打印 C++ 事件队列 (标准 JSON 输出)
// ========================================================
void consume_and_print_events(void* pipeline, PipelineState* state) {
    if (state->pending_bucket_events.empty() && state->pending_truck_events.empty() && state->pending_timeout_events.empty()) return;

    for (const auto& ev : state->pending_bucket_events) {
        std::cout << "{\"类型\": \"铲斗事件\", \"票号\": \"" << ev.ticket_id
                  << "\", \"总装车数\": " << ev.total_truck_count
                  << ", \"当前铲斗数\": " << ev.current_bucket_count
                  << ", \"卸料开始时间\": " << ev.dump_start_time
                  << ", \"卸料结束时间\": " << ev.dump_end_time << "}" << std::endl;
    }

    for (const auto& ev : state->pending_truck_events) {
        std::cout << "{\"类型\": \"装车完成事件\", \"票号\": \"" << ev.ticket_id
                  << "\", \"总装车数\": " << ev.total_truck_count
                  << ", \"总共铲斗数\": " << ev.total_bucket_count
                  << ", \"装车开始时间\": " << ev.load_start_time
                  << ", \"装车结束时间\": " << ev.load_end_time << "}" << std::endl;
    }

    for (const auto& ev : state->pending_timeout_events) {
        std::cout << "{\"类型\": \"车辆超时事件\", \"票号\": \"" << ev.ticket_id
                  << "\", \"备注\": \"已等待超限，未执行清理票号\"}" << std::endl;
    }

    clear_pipeline_events(pipeline);
}

// ========================================================
// 模式 1：常规视频流推理测试 
// ========================================================
void test_real_video(void* pipeline, cv::VideoCapture& cap, const char* out_pattern) {
    std::cout << "▶ 开始执行【模式 1: 常规真实视频流测试】..." << std::endl;
    cv::Mat frame;
    int frame_count = 0;
    long long total_time_us = 0;
    int perf_frame_count = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_count++;

        auto start_time = std::chrono::high_resolution_clock::now();
        process_frame(pipeline, frame.data, frame.cols, frame.rows, frame.channels());
        auto end_time = std::chrono::high_resolution_clock::now();
        
        total_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        perf_frame_count++;

        PipelineState* state = (PipelineState*)get_pipeline_state(pipeline);
        consume_and_print_events(pipeline, state);

        if (out_pattern != nullptr && frame_count % 5 == 0) {
            if (state->current_truck_box.area() > 0) {
                cv::rectangle(frame, state->current_truck_box, cv::Scalar(0, 255, 255), 2);
            }
            char filename[512];
            snprintf(filename, sizeof(filename), out_pattern, frame_count);
            cv::imwrite(filename, frame);
        }

        if (perf_frame_count == 100) {
            double avg_time_ms = (double)total_time_us / 1000.0 / 100.0;
            std::cout << "\n📊 [性能测试] 100帧平均耗时: " << avg_time_ms << " ms | FPS: " << (1000.0 / avg_time_ms) << "\n" << std::endl;
            total_time_us = 0; perf_frame_count = 0;
        }
    }
}

// ========================================================
// 模式 2：纯断电恢复测试
// ========================================================
void test_restore_only(void* pipeline, cv::VideoCapture& cap) {
    std::cout << "▶ 开始执行【模式 2: 纯断电恢复测试】..." << std::endl;

    std::cout << ">>> [动作] 模拟断电重启，灌入数据库状态：票号 TKT_RESTORED_888, 已装 5 铲, 矿物占比 0.65" << std::endl;
    restore_pipeline_state(pipeline, "TKT_RESTORED_888", 5, 0.65f);

    cv::Mat frame;
    int frame_count = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_count++;

        process_frame(pipeline, frame.data, frame.cols, frame.rows, frame.channels());
        PipelineState* state = (PipelineState*)get_pipeline_state(pipeline);
        consume_and_print_events(pipeline, state);

        if (frame_count > 40) {
            std::cout << "\n>>> [测试完成] 恢复流跑通，退出模拟。" << std::endl;
            break;
        }
    }
}

// ========================================================
// 模式 3：纯挂机超时预警测试
// ========================================================
// ========================================================
// 模式 3：纯挂机超时预警测试 (智能捕捉时机版)
// ========================================================
void test_timeout_only(void* pipeline, cv::VideoCapture& cap) {
    std::cout << "▶ 开始执行【模式 3: 纯挂机超时预警测试】..." << std::endl;

    std::cout << ">>> [动作] 为加快测试，设置系统判定超时时间为 10 秒" << std::endl;
    set_pipeline_timeout(pipeline, 10000);

    cv::Mat frame;
    int frame_count = 0;
    bool has_slept = false;
    int frames_after_sleep = 0;

    while (cap.read(frame)) {
        if (frame.empty()) break;
        frame_count++;

        process_frame(pipeline, frame.data, frame.cols, frame.rows, frame.channels());
        PipelineState* state = (PipelineState*)get_pipeline_state(pipeline);
        consume_and_print_events(pipeline, state);

        // 【核心修复】：不再傻等第 20 帧。只要发现它成功记上了至少 1 铲，立刻强行让司机去“吃饭”！
        if (!has_slept && state->current_truck_buckets > 0) {
            std::cout << "\n>>> ⏸ [模拟突发状况] 刚装进去 " << state->current_truck_buckets
                      << " 铲，司机突然去吃饭了，系统休眠 12 秒钟..." << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(12));

            std::cout << ">>> ▶ [状况解除] 挖机恢复运作，推入下一帧 (应该会立即触发单次超时推送！) \n" << std::endl;
            has_slept = true;
        }

        // 休眠结束后，再让视频往后跑 10 帧，看看系统有没有把超时事件吐出来
        if (has_slept) {
            frames_after_sleep++;
            if (frames_after_sleep > 10) {
                std::cout << "\n>>> [测试完成] 超时验证结束，退出模拟。" << std::endl;
                break;
            }
        }
    }
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <yolo.rknn> <in_video.mp4> <test_mode(1/2/3)> [out_pattern.jpg]" << std::endl;
        std::cerr << "test_mode 1: 常规视频推理" << std::endl;
        std::cerr << "test_mode 2: 断电恢复测试" << std::endl;
        std::cerr << "test_mode 3: 挂机超时测试" << std::endl;
        return -1;
    }

    const char* yolo_model_path = argv[1];
    const char* input_video = argv[2];
    int test_mode = std::atoi(argv[3]);
    const char* output_pattern = (argc > 4) ? argv[4] : nullptr;

    std::cout << "--> 加载 YOLO 模型..." << std::endl;
    auto yolo_buf = load_file_to_memory(yolo_model_path);
    void* pipeline = init_pipeline_from_memory(yolo_buf.data(), yolo_buf.size());
    if (!pipeline) return -1;

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频文件!" << std::endl;
        return -1;
    }

    // 根据传入的 mode 参数分流
    if (test_mode == 1) {
        test_real_video(pipeline, cap, output_pattern);
    } else if (test_mode == 2) {
        test_restore_only(pipeline, cap);
    } else if (test_mode == 3) {
        test_timeout_only(pipeline, cap);
    } else {
        std::cerr << "❌ 未知的 test_mode! 请输入 1, 2 或 3。" << std::endl;
    }

    cap.release();
    release_pipeline(pipeline);
    return 0;
}