#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "excavator_pipeline.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "用法: " << argv[0] << " <yolo.rknn> <siamese.rknn> <in_video.mp4> [out_video.mp4]" << std::endl;
        return -1;
    }

    const char* yolo_model = argv[1];
    const char* siamese_model = argv[2];
    const char* input_video = argv[3];

    // 如果给了第四个参数，就生成视频，否则传 nullptr 不渲染硬盘写入
    const char* output_video = (argc > 4) ? argv[4] : nullptr;

    std::cout << "--> 正在初始化 NPU 双模型流水线引擎..." << std::endl;
    void* pipeline = init_pipeline(yolo_model, siamese_model, output_video);
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

        // 记录推理起始时间
        auto start = std::chrono::high_resolution_clock::now();

        // 核心推理与渲染都在这句完成了
        process_frame(pipeline, frame.data, frame.cols, frame.rows, frame.channels());

        // 记录推理结束时间
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        total_time_ms += elapsed_ms;
        frame_count++;

        // 实时打印进度（每 10 帧打印一次，包含 std::endl 强制刷新控制台缓冲区）
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