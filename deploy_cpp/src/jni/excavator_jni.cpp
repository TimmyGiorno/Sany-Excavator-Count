#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/opencv.hpp>
#include "excavator_pipeline.h"

// ================= 宏定义与日志打印 =================
#define LOG_TAG "ExcavatorJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// ================= 内存映射结构体 =================
struct PipelineState {
    std::string ticket_id;             // 新增：票号
    int total_truck_count;
    int total_bucket_count;            // 绝对总斗数
    bool bucket_full;
    bool dumping_active;
    cv::Rect last_dumping_box;
    cv::Mat reference_truck_img;       // 替换旧变量
    std::vector<float> reference_truck_emb; // 替换旧变量

    cv::Rect current_bucket_box;
    int current_bucket_type;
    cv::Rect current_truck_box;
    bool is_new_truck_entered;
};

// ================= 辅助函数区 =================

// 辅助函数：触发 Java 的 InitCallback 回调
void triggerInitCallback(JNIEnv *env, jobject callback, bool success, const char* errorMsg = "") {
    if (!callback) return;
    jclass callbackClass = env->GetObjectClass(callback);
    if (success) {
        jmethodID onSuccessMethod = env->GetMethodID(callbackClass, "onSuccess", "()V");
        env->CallVoidMethod(callback, onSuccessMethod);
    } else {
        jmethodID onFailureMethod = env->GetMethodID(callbackClass, "onFailure", "(Ljava/lang/String;)V");
        jstring jErrorMsg = env->NewStringUTF(errorMsg);
        env->CallVoidMethod(callback, onFailureMethod, jErrorMsg);
        env->DeleteLocalRef(jErrorMsg);
    }
    env->DeleteLocalRef(callbackClass);
}

// 辅助函数：读取本地文件到内存
unsigned char* loadFileToMemory(const char* filepath, int* out_size) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);
    *out_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* buffer = (unsigned char*)malloc(*out_size);
    if (*out_size != fread(buffer, 1, *out_size, fp)) {
        free(buffer);
        fclose(fp);
        return nullptr;
    }
    fclose(fp);
    return buffer;
}


extern "C" {

// ================= 接口 1：从文件路径加载 =================
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromFile(JNIEnv *env, jclass clazz, jstring yoloPath, jstring siamesePath, jobject callback) {
    const char *yolo_path_c = env->GetStringUTFChars(yoloPath, 0);
    const char *siamese_path_c = env->GetStringUTFChars(siamesePath, 0);

    int yolo_size = 0, siamese_size = 0;
    unsigned char* yolo_buf = loadFileToMemory(yolo_path_c, &yolo_size);
    unsigned char* siamese_buf = loadFileToMemory(siamese_path_c, &siamese_size);

    if (!yolo_buf || !siamese_buf) {
        LOGE("Failed to load models from SD Card.");
        triggerInitCallback(env, callback, false, "Failed to read model files from path");
    } else {
        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size, siamese_buf, siamese_size);
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));

        LOGI("Models loaded successfully from SD Card.");
        triggerInitCallback(env, callback, true);
    }

    if (yolo_buf) free(yolo_buf);
    if (siamese_buf) free(siamese_buf);
    env->ReleaseStringUTFChars(yoloPath, yolo_path_c);
    env->ReleaseStringUTFChars(siamesePath, siamese_path_c);
}

// ================= 接口 2：从 Asset 零拷贝加载 =================
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromAsset(JNIEnv *env, jclass clazz, jobject assetManager, jstring yoloFileName, jstring siameseFileName, jobject callback) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    const char *yolo_name_c = env->GetStringUTFChars(yoloFileName, 0);
    const char *siamese_name_c = env->GetStringUTFChars(siameseFileName, 0);

    AAsset* yolo_asset = AAssetManager_open(mgr, yolo_name_c, AASSET_MODE_BUFFER);
    AAsset* siamese_asset = AAssetManager_open(mgr, siamese_name_c, AASSET_MODE_BUFFER);

    if (!yolo_asset || !siamese_asset) {
        LOGE("Failed to open models from Assets.");
        triggerInitCallback(env, callback, false, "Failed to open model files from assets");
    } else {
        const void* yolo_buf = AAsset_getBuffer(yolo_asset);
        off_t yolo_size = AAsset_getLength(yolo_asset);
        const void* siamese_buf = AAsset_getBuffer(siamese_asset);
        off_t siamese_size = AAsset_getLength(siamese_asset);

        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size, siamese_buf, siamese_size);
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));

        LOGI("Models loaded successfully from Assets.");
        triggerInitCallback(env, callback, true);
    }

    if (yolo_asset) AAsset_close(yolo_asset);
    if (siamese_asset) AAsset_close(siamese_asset);
    env->ReleaseStringUTFChars(yoloFileName, yolo_name_c);
    env->ReleaseStringUTFChars(siameseFileName, siamese_name_c);
}

// ================= 接口 3：从内存字节数组加载 =================
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromByteArray(JNIEnv *env, jclass clazz, jbyteArray yoloData, jbyteArray siameseData, jobject callback) {
    jsize yolo_size = env->GetArrayLength(yoloData);
    jbyte* yolo_buf = env->GetByteArrayElements(yoloData, NULL);

    jsize siamese_size = env->GetArrayLength(siameseData);
    jbyte* siamese_buf = env->GetByteArrayElements(siameseData, NULL);

    if (yolo_size == 0 || siamese_size == 0) {
        triggerInitCallback(env, callback, false, "Byte array is empty");
    } else {
        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size, siamese_buf, siamese_size);
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));

        LOGI("Models loaded successfully from Byte Arrays.");
        triggerInitCallback(env, callback, true);
    }

    env->ReleaseByteArrayElements(yoloData, yolo_buf, JNI_ABORT);
    env->ReleaseByteArrayElements(siameseData, siamese_buf, JNI_ABORT);
}

// ================= 接口 4：视频流检测与业务回调 =================
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_detectNative(JNIEnv *env, jclass clazz, jlong handlePtr, jbyteArray yuvData, jint width, jint height, jobject callback) {
    if (handlePtr == 0 || !yuvData || !callback) return;

    jbyte* yuv_buf = env->GetByteArrayElements(yuvData, NULL);
    cv::Mat yuv(height + height / 2, width, CV_8UC1, (unsigned char*)yuv_buf);
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);
    env->ReleaseByteArrayElements(yuvData, yuv_buf, JNI_ABORT);

    void* handle = reinterpret_cast<void*>(handlePtr);
    process_frame(handle, bgr.data, bgr.cols, bgr.rows, 3);

    PipelineState* state = (PipelineState*)get_pipeline_state(handle);
    if (!state) return;

    jclass resultClass = env->FindClass("com/rosenshine/hhd/Excavator/ExcavatorResult");
    jmethodID constructor = env->GetMethodID(resultClass, "<init>", "()V");
    jobject resultObj = env->NewObject(resultClass, constructor);

    // ============= 填充基础业务字段 =============
    env->SetIntField(resultObj, env->GetFieldID(resultClass, "currentShovelCount", "I"), state->total_bucket_count); // 传绝对斗数，Java去减
    env->SetBooleanField(resultObj, env->GetFieldID(resultClass, "isLoading", "Z"), state->dumping_active);
    env->SetBooleanField(resultObj, env->GetFieldID(resultClass, "isComplete", "Z"), state->is_new_truck_entered);
    env->SetBooleanField(resultObj, env->GetFieldID(resultClass, "isStartLoading", "Z"), state->total_bucket_count >= 1);
    env->SetIntField(resultObj, env->GetFieldID(resultClass, "bucketType", "I"), state->current_bucket_type);

    // 【新增】处理 C++ 字符串到 Java 字符串的转换
    jstring jTicketId = env->NewStringUTF(state->ticket_id.c_str());
    env->SetObjectField(resultObj, env->GetFieldID(resultClass, "ticketId", "Ljava/lang/String;"), jTicketId);
    env->DeleteLocalRef(jTicketId);

    if (!state->reference_truck_emb.empty()) {
        int featureSize = state->reference_truck_emb.size();
        jfloatArray featureArray = env->NewFloatArray(featureSize);
        env->SetFloatArrayRegion(featureArray, 0, featureSize, state->reference_truck_emb.data());
        env->SetObjectField(resultObj, env->GetFieldID(resultClass, "truckFeature", "[F"), featureArray);
        env->DeleteLocalRef(featureArray);
    }

    // 填充 Rect (保持不变)
    jclass rectClass = env->FindClass("android/graphics/Rect");
    jfieldID leftField = env->GetFieldID(rectClass, "left", "I");
    jfieldID topField = env->GetFieldID(rectClass, "top", "I");
    jfieldID rightField = env->GetFieldID(rectClass, "right", "I");
    jfieldID bottomField = env->GetFieldID(rectClass, "bottom", "I");

    jobject bucketRect = env->GetObjectField(resultObj, env->GetFieldID(resultClass, "bucketPosition", "Landroid/graphics/Rect;"));
    env->SetIntField(bucketRect, leftField, state->current_bucket_box.x);
    env->SetIntField(bucketRect, topField, state->current_bucket_box.y);
    env->SetIntField(bucketRect, rightField, state->current_bucket_box.x + state->current_bucket_box.width);
    env->SetIntField(bucketRect, bottomField, state->current_bucket_box.y + state->current_bucket_box.height);

    jobject truckRect = env->GetObjectField(resultObj, env->GetFieldID(resultClass, "truckPosition", "Landroid/graphics/Rect;"));
    env->SetIntField(truckRect, leftField, state->current_truck_box.x);
    env->SetIntField(truckRect, topField, state->current_truck_box.y);
    env->SetIntField(truckRect, rightField, state->current_truck_box.x + state->current_truck_box.width);
    env->SetIntField(truckRect, bottomField, state->current_truck_box.y + state->current_truck_box.height);

    // 触发回调
    jclass cbClass = env->GetObjectClass(callback);
    jmethodID onResultMethod = env->GetMethodID(cbClass, "onResult", "(Lcom/rosenshine/hhd/Excavator/ExcavatorResult;)V");
    env->CallVoidMethod(callback, onResultMethod, resultObj);

    env->DeleteLocalRef(resultClass);
    env->DeleteLocalRef(resultObj);
    env->DeleteLocalRef(rectClass);
    env->DeleteLocalRef(bucketRect);
    env->DeleteLocalRef(truckRect);
    env->DeleteLocalRef(cbClass);
}

// ================= 3. 新增动态调参接口 =================
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_updateConfigNative(JNIEnv *env, jclass clazz, jlong handlePtr, jfloat confThresh, jfloat iouThresh, jfloat siameseThresh) {
    if (handlePtr != 0) {
        update_pipeline_config(reinterpret_cast<void*>(handlePtr), confThresh, iouThresh, siameseThresh);
    }
}

// ================= 接口 5：安全释放 NPU 与内存池 =================
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_releaseNative(JNIEnv *env, jclass clazz, jlong handlePtr) {
    if (handlePtr != 0) {
        void* handle = reinterpret_cast<void*>(handlePtr);
        release_pipeline(handle);

        // 重置 Java 层的指针引用
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, 0);

        LOGI("Hardware pipeline resources released completely.");
    }
}

// ================= 新增：从 Java 接收断电恢复数据 =================
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_restoreStateNative(JNIEnv *env, jclass clazz, jlong handlePtr, jstring ticketId, jint bucketCount, jfloatArray featureArray) {
    if (handlePtr == 0 || !featureArray) return;

    const char* t_id = env->GetStringUTFChars(ticketId, 0);
    int feature_len = env->GetArrayLength(featureArray);
    jfloat* feature_data = env->GetFloatArrayElements(featureArray, NULL);

    // 灌入底层 C++
    restore_pipeline_state(reinterpret_cast<void*>(handlePtr), t_id, bucketCount, feature_data, feature_len);

    // 释放内存锁
    env->ReleaseFloatArrayElements(featureArray, feature_data, JNI_ABORT);
    env->ReleaseStringUTFChars(ticketId, t_id);
    LOGI("Hardware Pipeline State Restored: %s, Buckets: %d", t_id, bucketCount);
}

} // extern "C"