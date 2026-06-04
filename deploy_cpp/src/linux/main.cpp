#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "excavator_pipeline.h"

// ========================================================
// 偷天换日：在测试程序里声明一个和算法层长得一模一样的结构体
// 这样我们就能把底层的 void* 强转成可视化的状态对象，方便画图
// ========================================================
struct PipelineState {
    int total_truck_count;
    int total_bucket_count;
    bool bucket_full;
    bool dumping_active;
    cv::Rect last_dumping_box;
    cv::Mat last_truck_img;
    cv::Mat previous_truck_img;
    std::vector<float> last_truck_emb;

    cv::Rect current_bucket_box;
    int current_bucket_type;  // -1无, 0空, 1满
    cv::Rect current_truck_box;
    bool is_new_truck_entered;
};

// 辅助函数：把本地文件一次性读进内存数组
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

    // 如果给了第四个参数，代表我们要输出图片序列（比如 /data/local/tmp/out_frames/frame_%04d.jpg）
    const char* output_pattern = (argc > 4) ? argv[4] : nullptr;

    std::cout << "--> 正在将 RKNN 模型文件加载进系统内存..." << std::endl;
    auto yolo_buf = load_file_to_memory(yolo_model_path);
    auto siamese_buf = load_file_to_memory(siamese_model_path);

    if (yolo_buf.empty() || siamese_buf.empty()) {
        std::cerr << "❌ 模型加载失败，请检查文件路径！" << std::endl;
        return -1;
    }

    std::cout << "--> 正在初始化 NPU 双模型流水线引擎..." << std::endl;
    // 使用新的基于内存的初始化接口
    void* pipeline = init_pipeline_from_memory(yolo_buf.data(), yolo_buf.size(), siamese_buf.data(), siamese_buf.size());
    if (!pipeline) {
        std::cerr << "❌ 流水线初始化失败！" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频源文件: " << input_video << std::endl;
        release_pipeline(pipeline);
        return -1;
    }

    std::cout << "--> 开始解析视频流并送入 NPU 处理..." << std::endl;

    int frame_count = 0;
    double total_time_ms = 0.0;
    cv::Mat frame;

    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cout << "--> 视频流读取完毕。" << std::endl;
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // 1. 核心推理 (纯净版，不包含画图)
        process_frame(pipeline, frame.data, frame.cols, frame.rows, frame.channels());

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        total_time_ms += elapsed_ms;
        frame_count++;

        // 2. 获取底层算法推断出的状态 (完全模拟 Java 回调时拿数据的过程)
        PipelineState* state = (PipelineState*)get_pipeline_state(pipeline);

        // 3. 在外层执行可视化渲染 (把画图的脏活留给应用层)
        if (output_pattern != nullptr && state != nullptr) {
            // 画卡车
            if (state->current_truck_box.area() > 0) {
                cv::rectangle(frame, state->current_truck_box, cv::Scalar(0, 255, 255), 2);
                cv::putText(frame, "Truck", cv::Point(state->current_truck_box.x, state->current_truck_box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
            }

            // 画铲斗
            if (state->current_bucket_type != -1) {
                cv::Scalar color = (state->current_bucket_type == 1) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
                std::string label = (state->current_bucket_type == 1) ? "Bucket-Full" : "Bucket-Empty";
                cv::rectangle(frame, state->current_bucket_box, color, 2);
                cv::putText(frame, label, cv::Point(state->current_bucket_box.x, state->current_bucket_box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }

            // 画卸料状态指示灯
            if (state->dumping_active) {
                cv::putText(frame, "STATUS: DUMPING", cv::Point(30, 130), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 255), 2);
            }

            // 画左上角计数看板
            cv::rectangle(frame, cv::Point(15, 15), cv::Point(250, 100), cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, "Trucks: " + std::to_string(state->total_truck_count), cv::Point(25, 45),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, "Buckets: " + std::to_string(state->total_bucket_count), cv::Point(25, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            // 格式化输出路径并保存当前帧
            char filename[512];
            snprintf(filename, sizeof(filename), output_pattern, frame_count);
            cv::imwrite(filename, frame);
        }

        // 实时打印进度
        if (frame_count % 10 == 0) {
            double current_fps = 1000.0 / elapsed_ms;
            double avg_fps = 1000.0 / (total_time_ms / frame_count);
            std::cout << "帧号: " << frame_count
                      << " | 单帧耗时: " << elapsed_ms << " ms"
                      << " | 瞬时 FPS: " << current_fps
                      << " | 平均 FPS: " << avg_fps << std::endl;
        }
    }

    // 循环结束，输出总结报告
    std::cout << "\n================ 性能分析报告 ================" << std::endl;
    std::cout << " 📊 总处理帧数: " << frame_count << " 帧" << std::endl;
    if (frame_count > 0) {
        std::cout << " ⏱️ 平均单帧耗时: " << (total_time_ms / frame_count) << " ms" << std::endl;
        std::cout << " 🚀 综合推理速度: " << 1000.0 / (total_time_ms / frame_count) << " FPS" << std::endl;
    }
    std::cout << "=============================================\n" << std::endl;

    cap.release();
    release_pipeline(pipeline);
    std::cout << "--> 内存池及 NPU 硬件资源已安全释放。" << std::endl;

    return 0;
}