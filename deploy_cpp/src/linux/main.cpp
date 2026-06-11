#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "excavator_pipeline.h"

// ========================================================
// 结构体必须和 excavator_pipeline.cpp 中的 PipelineState 完全对齐！
// 否则强转 void* 时会导致严重的内存越界崩溃。
// ========================================================
struct PipelineState {
    std::string ticket_id;             // 新增：票号
    int total_truck_count;
    int total_bucket_count;            // 绝对全局斗数
    bool bucket_full;
    bool dumping_active;
    cv::Rect last_dumping_box;
    cv::Mat reference_truck_img;       // 替换旧变量
    std::vector<float> reference_truck_emb; // 替换旧变量

    cv::Rect current_bucket_box;
    int current_bucket_type;
    cv::Rect current_truck_box;
    bool is_new_truck_entered;         // 新增：事件触发器
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
        fclose(fp);
        return {};
    }
    fclose(fp);
    return buffer;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <yolo.rknn> <siamese.rknn> <in_video.mp4> [out_frame_pattern.jpg]" << std::endl;
        return -1;
    }

    const char* yolo_model_path = argv[1];
    const char* siamese_model_path = argv[2];
    const char* input_video = argv[3];
    const char* output_pattern = (argc > 4) ? argv[4] : nullptr;

    std::cout << "--> 加载 RKNN 模型..." << std::endl;
    auto yolo_buf = load_file_to_memory(yolo_model_path);
    auto siamese_buf = load_file_to_memory(siamese_model_path);

    void* pipeline = init_pipeline_from_memory(yolo_buf.data(), yolo_buf.size(), siamese_buf.data(), siamese_buf.size());
    if (!pipeline) return -1;

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) return -1;

    int frame_count = 0;
    double total_time_ms = 0.0;
    cv::Mat frame;

    // ========================================================
    // 模拟 Java 业务层的状态变量
    // ========================================================
    int base_bucket_count = 0;
    int current_truck_buckets = 0;
    std::string current_active_ticket = "";

    while (cap.read(frame)) {
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();
        process_frame(pipeline, frame.data, frame.cols, frame.rows, frame.channels());
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        total_time_ms += elapsed_ms;
        frame_count++;

        PipelineState* state = (PipelineState*)get_pipeline_state(pipeline);

        if (output_pattern != nullptr && state != nullptr) {

            // ========================================================
            // 业务层逻辑：计算每车斗数
            // ========================================================
            if (state->is_new_truck_entered) {
                current_active_ticket = state->ticket_id;
                // 换车时，记录此时的绝对总斗数作为这辆车的基数
                base_bucket_count = state->total_bucket_count - 1;
                std::cout << ">>> [业务事件] 换车/新入场! 票号: " << current_active_ticket << std::endl;
            }

            // 当前这辆车的实际斗数 = 全局绝对斗数 - 这辆车入场时的基数
            if (!current_active_ticket.empty()) {
                current_truck_buckets = state->total_bucket_count - base_bucket_count;
            }

            // ========== OpenCV 画面渲染 ==========
            if (state->current_truck_box.area() > 0) {
                cv::rectangle(frame, state->current_truck_box, cv::Scalar(0, 255, 255), 2);
            }
            if (state->current_bucket_type != -1) {
                cv::Scalar color = (state->current_bucket_type == 1) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                cv::rectangle(frame, state->current_bucket_box, color, 2);
            }

            // UI 黑色底板加宽，以容纳票号
            cv::rectangle(frame, cv::Point(15, 15), cv::Point(400, 130), cv::Scalar(0, 0, 0), cv::FILLED);

            // 绘制卡车总数
            cv::putText(frame, "Trucks: " + std::to_string(state->total_truck_count), cv::Point(25, 45),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            // 绘制【当前这辆车的】斗数
            cv::putText(frame, "Cur Buckets: " + std::to_string(current_truck_buckets), cv::Point(25, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            // 绘制票号
            if (!state->ticket_id.empty()) {
                cv::putText(frame, "Ticket: " + state->ticket_id, cv::Point(25, 115),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }

            char filename[512];
            snprintf(filename, sizeof(filename), output_pattern, frame_count);
            cv::imwrite(filename, frame);
        }

        if (frame_count % 10 == 0) {
            std::cout << "帧号: " << frame_count << " | 耗时: " << elapsed_ms << " ms" << std::endl;
        }
    }

    cap.release();
    release_pipeline(pipeline);
    return 0;
}