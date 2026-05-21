#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>


class YolovUltralyticsInference {
public:
    std::vector<std::string> labels;
    std::string onnx_path_name;
    cv::Mat input_image, result_image;
    float confidence_thres;
    float iou_thres;
    int model_input_w, model_input_h, model_output_h, model_output_w;
    float x_factor, y_factor, ratio;
    std::vector<std::string> preprocess_method;
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    int top, left;

    YolovUltralyticsInference(
        std::vector<std::string> labels,
        std::string onnx_path_name,
        std::vector<std::string> preprocess_method = std::vector<std::string>{"letter_box"},
        float confidence_thres = 0.25,
        float iou_thres = 0.45)
        : labels(labels), onnx_path_name(onnx_path_name),
          preprocess_method(preprocess_method), confidence_thres(confidence_thres),
          iou_thres(iou_thres), env(ORT_LOGGING_LEVEL_ERROR, "yolov-onnx"), session_options(),
          session(nullptr)
    {
        this->model_input_w = 0;
        this->model_input_h = 0;
        this->model_output_h = 0;
        this->model_output_w = 0;
        this->x_factor = 0;
        this->y_factor = 0;

        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        read_model();
    }

    void read_model() {
        std::ifstream infile(onnx_path_name);
        if (!infile.good()) {
            throw std::runtime_error("找不到 ONNX 模型文件，请检查文件名和路径: " + onnx_path_name);
        }

// 针对 Windows 环境强制转换宽字符 (wstring) 路径
#ifdef _WIN32
        std::wstring widestr = std::wstring(onnx_path_name.begin(), onnx_path_name.end());
        session = Ort::Session(env, widestr.c_str(), session_options);
#else
        session = Ort::Session(env, onnx_path_name.c_str(), session_options);
#endif

        size_t numInputNodes = session.GetInputCount();
        size_t numOutputNodes = session.GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < numInputNodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();

            this->model_input_w = input_dims[3];
            this->model_input_h = input_dims[2];
        }

        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        this->model_output_h = output_dims[1];
        this->model_output_w = output_dims[2];

        for (size_t i = 0; i < numOutputNodes; i++) {
            auto out_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(out_name.get());
        }
    }

    cv::Mat preprocess() {
        this->result_image = this->input_image.clone();
        cv::Mat blob;
        cv::cvtColor(this->input_image, this->input_image, cv::COLOR_BGR2RGB);

        if (this->preprocess_method == std::vector<std::string>{"letter_box"}) {
            float ratio = std::min(static_cast<float>(this->model_input_h) / this->input_image.rows,
                                   static_cast<float>(this->model_input_w) / this->input_image.cols);
            int newh = (int)std::round(this->input_image.rows * ratio);
            int neww = (int)std::round(this->input_image.cols * ratio);
            cv::Size new_unpad(neww, newh);

            float dw = (this->model_input_w - neww) / 2.0f;
            float dh = (this->model_input_h - newh) / 2.0f;

            if (neww != this->model_input_w || newh != this->model_input_h) {
                cv::resize(this->input_image, this->input_image, new_unpad, cv::INTER_LINEAR);
            }

            int top = (int)std::round(dh - 0.1);
            int bottom = (int)std::round(dh + 0.1);
            int left = (int)std::round(dw - 0.1);
            int right = (int)std::round(dw + 0.1);

            this->top = top;
            this->left = left;
            this->ratio = ratio;

            cv::copyMakeBorder(this->input_image, this->input_image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            cv::dnn::blobFromImage(this->input_image, blob, 1 / 255.0, cv::Size(this->model_input_w, this->model_input_h), cv::Scalar(0, 0, 0), true, false);
        } else {
            cv::resize(this->input_image, this->input_image, cv::Size(this->model_input_w, this->model_input_h));
            cv::dnn::blobFromImage(this->input_image, blob, 1 / 255.0, cv::Size(), cv::Scalar(0, 0, 0), false, false);
        }
        return blob;
    }

    cv::Mat draw_detections(cv::Mat img, std::vector<int> indexes, std::vector<cv::Rect> boxes, std::vector<int> classIds) {
        for (size_t i = 0; i < indexes.size(); i++) {
            int index = indexes[i];
            int class_id = classIds[index];

            cv::Scalar color = (class_id == 2 || class_id == 3) ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 165, 255);
            std::string label = this->labels[class_id];

            cv::rectangle(img, boxes[index], color, 2, 8);
            cv::putText(img, label, cv::Point(boxes[index].x, boxes[index].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        return img;
    }

    cv::Mat main_process(cv::Mat current_frame) {
        this->input_image = current_frame;
        this->x_factor = this->input_image.cols / static_cast<float>(this->model_input_w);
        this->y_factor = this->input_image.rows / static_cast<float>(this->model_input_h);

        cv::Mat blob = this->preprocess();
        size_t tpixels = this->model_input_h * this->model_input_w * 3;
        std::array<int64_t, 4> input_shape_info{1, 3, this->model_input_h, this->model_input_w};

        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

        const std::array<const char*, 1> inputNames = {this->input_node_names[0].c_str()};
        const std::array<const char*, 1> outNames = {this->output_node_names[0].c_str()};

        std::vector<Ort::Value> ort_outputs;
        ort_outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &input_tensor, 1, outNames.data(), outNames.size());

        const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
        cv::Mat dout(this->model_output_h, this->model_output_w, CV_32F, (float*)pdata);
        cv::Mat det_output = dout.t();

        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < det_output.rows; i++) {
            cv::Mat classes_scores = det_output.row(i).colRange(4, 4 + this->labels.size());
            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, nullptr, &score, nullptr, &classIdPoint);

            if (score > this->confidence_thres) {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);

                if (this->preprocess_method == std::vector<std::string>{"letter_box"}) {
                    cx = (cx - this->left) / this->ratio;
                    cy = (cy - this->top) / this->ratio;
                    ow = ow / this->ratio;
                    oh = oh / this->ratio;
                }

                int x = static_cast<int>(cx - 0.5 * ow);
                int y = static_cast<int>(cy - 0.5 * oh);
                cv::Rect box(x, y, static_cast<int>(ow), static_cast<int>(oh));

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }

        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, this->confidence_thres, this->iou_thres, indexes);
        this->draw_detections(this->result_image, indexes, boxes, classIds);

        return result_image;
    }
};

int main() {
    setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    system("chcp 65001 > nul");

    std::cout << ">>> 启动 ONNXRuntime 视频测试..." << std::endl;

    std::vector<std::string> labels = {"bucket-empty", "bucket-full", "truck-empty", "truck-full"};
    std::string video_name = "test_video.mp4";
    std::string onnx_path_name = "best.onnx";
    std::vector<std::string> preprocess_method = std::vector<std::string>{"letter_box"};

    std::cout << ">>> 正在加载模型: " << onnx_path_name << std::endl;
    YolovUltralyticsInference inference(labels, onnx_path_name, preprocess_method);
    std::cout << ">>> 模型加载成功！" << std::endl;

    cv::VideoCapture cap(video_name);
    if (!cap.isOpened()) {
        std::cerr << "找不到测试视频，请确认 " << video_name << " 是否放在 exe 同级目录下！" << std::endl;
        return -1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    cv::VideoWriter writer("result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    cv::Mat frame;
    int frame_count = 0;

    std::cout << ">>> 正在处理视频，总帧数: " << total_frames << " ..." << std::endl;

    while (cap.read(frame) && frame_count < 500) {
        frame_count++;
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat output_frame = inference.main_process(frame);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (duration > 0) {
            std::string fps_text = "FPS: " + std::to_string(1000 / duration) + " (" + std::to_string(duration) + "ms)";
            cv::putText(output_frame, fps_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        writer.write(output_frame);

        if (frame_count % 30 == 0) {
            std::cout << "处理进度: " << frame_count << " / " << total_frames << " 帧" << std::endl;
        }
    }

    cap.release();
    writer.release();
    std::cout << ">>> 视频处理完成！结果已保存至 result.mp4" << std::endl;

    return 0;
}