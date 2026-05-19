#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>

#include "../include/inference_engine.h"

/* ---- Backend factory (picked at compile time by CMake defines) ---- */

#ifdef USE_ONNX
IBackend* create_onnx_backend();
#define BACKEND_TAG "ONNX"
#endif

#ifdef USE_MOCK
IBackend* create_mock_backend();
#define BACKEND_TAG "Mock"
#endif

/* ---- Main ---- */

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    std::string video_path = (argc > 1) ? argv[1] : "test_video.mp4";
    std::string model_path = (argc > 2) ? argv[2] : "best.onnx";

    std::cout << "=== Excavator Detector [" BACKEND_TAG "] ===" << std::endl;
    std::cout << "Video: " << video_path << std::endl;
    std::cout << "Model: " << model_path << std::endl;

    /* ---- Init engine ---- */
    InferenceEngine engine;
    IBackend* backend = nullptr;

#ifdef USE_ONNX
    backend = create_onnx_backend();
#else
    backend = create_mock_backend();
#endif

    if (!engine.load_model(backend, model_path.c_str())) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }
    std::cout << "Model loaded. input=" << engine.input_width()
              << "x" << engine.input_height() << std::endl;

    /* ---- Open video ---- */
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << std::endl;
        return -1;
    }

    int vw = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int vh = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    cv::VideoWriter writer("result.mp4",
        cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(vw, vh));

    const char* names[] = {"bucket-empty","bucket-full","truck-empty","truck-full"};

    cv::Mat frame;
    int count = 0;

    std::cout << "Processing " << total << " frames..." << std::endl;

    while (cap.read(frame)) {
        if (count >= 500) break;
        count++;

        auto t0 = std::chrono::high_resolution_clock::now();

        cv::Mat rgb;
        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

        std::vector<DetectBox> boxes = engine.detect(rgb.data, rgb.cols, rgb.rows);

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        for (const auto& b : boxes) {
            cv::Scalar color = (b.class_id == 2 || b.class_id == 3)
                ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 165, 255);
            cv::rectangle(frame,
                cv::Point((int)b.x1, (int)b.y1),
                cv::Point((int)b.x2, (int)b.y2), color, 2);
            const char* label = (b.class_id >= 0 && b.class_id < 4)
                ? names[b.class_id] : "?";
            cv::putText(frame, label,
                cv::Point((int)b.x1, (int)b.y1 - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        if (ms > 0) {
            std::string fps_text = "FPS: " + std::to_string(1000 / ms)
                + " (" + std::to_string(ms) + "ms)";
            cv::putText(frame, fps_text, cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }

        writer.write(frame);

        if (count % 30 == 0) {
            std::cout << count << "/" << total
                      << "  boxes=" << boxes.size()
                      << "  " << ms << "ms/f" << std::endl;
        }
    }

    cap.release();
    writer.release();
    std::cout << "Done. " << count << " frames -> result.mp4" << std::endl;
    return 0;
}
