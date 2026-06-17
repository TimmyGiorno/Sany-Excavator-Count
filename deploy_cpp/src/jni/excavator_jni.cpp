#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <opencv2/opencv.hpp>
#include "excavator_pipeline.h"

#define LOG_TAG "ExcavatorJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

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

unsigned char* loadFileToMemory(const char* filepath, int* out_size) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) return nullptr;
    fseek(fp, 0, SEEK_END);
    *out_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char* buffer = (unsigned char*)malloc(*out_size);
    if (*out_size != fread(buffer, 1, *out_size, fp)) {
        free(buffer); fclose(fp);
        return nullptr;
    }
    fclose(fp);
    return buffer;
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromFile(JNIEnv *env, jclass clazz, jstring yoloPath, jobject callback) {
    const char *yolo_path_c = env->GetStringUTFChars(yoloPath, 0);
    int yolo_size = 0;
    unsigned char* yolo_buf = loadFileToMemory(yolo_path_c, &yolo_size);

    if (!yolo_buf) {
        triggerInitCallback(env, callback, false, "Failed to read YOLO model");
    } else {
        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size);
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));
        triggerInitCallback(env, callback, true);
    }
    if (yolo_buf) free(yolo_buf);
    env->ReleaseStringUTFChars(yoloPath, yolo_path_c);
}

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromAsset(JNIEnv *env, jclass clazz, jobject assetManager, jstring yoloFileName, jobject callback) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    const char *yolo_name_c = env->GetStringUTFChars(yoloFileName, 0);
    AAsset* yolo_asset = AAssetManager_open(mgr, yolo_name_c, AASSET_MODE_BUFFER);

    if (!yolo_asset) {
        triggerInitCallback(env, callback, false, "Failed to open YOLO model");
    } else {
        const void* yolo_buf = AAsset_getBuffer(yolo_asset);
        off_t yolo_size = AAsset_getLength(yolo_asset);
        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size);
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));
        triggerInitCallback(env, callback, true);
    }
    if (yolo_asset) AAsset_close(yolo_asset);
    env->ReleaseStringUTFChars(yoloFileName, yolo_name_c);
}

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromByteArray(JNIEnv *env, jclass clazz, jbyteArray yoloData, jobject callback) {
    jsize yolo_size = env->GetArrayLength(yoloData);
    jbyte* yolo_buf = env->GetByteArrayElements(yoloData, NULL);

    if (yolo_size == 0) {
        triggerInitCallback(env, callback, false, "YOLO Byte array empty");
    } else {
        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size);
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));
        triggerInitCallback(env, callback, true);
    }
    env->ReleaseByteArrayElements(yoloData, yolo_buf, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_detectNative(JNIEnv *env, jclass clazz, jlong handlePtr, jbyteArray yuvData, jint width, jint height, jobject callback) {
    if (handlePtr == 0 || !yuvData || !callback) return;

    jbyte* yuv_buf = env->GetByteArrayElements(yuvData, NULL);
    cv::Mat yuv(height + height / 2, width, CV_8UC1, (unsigned char*)yuv_buf);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);
    env->ReleaseByteArrayElements(yuvData, yuv_buf, JNI_ABORT);

    void* handle = reinterpret_cast<void*>(handlePtr);

    process_frame(handle, bgr.data, bgr.cols, bgr.rows, 3);

    // ================= 严格对齐最新状态机的内部结构 =================
    struct BucketEvent { std::string ticket_id; int total_truck_count; int current_bucket_count; long long dump_start_time; long long dump_end_time; float last_mineral_ratio; };
    struct TruckEvent { std::string ticket_id; int total_truck_count; int total_bucket_count; long long load_start_time; long long load_end_time; int completed_type; };
    struct PendingBucket { long long dump_start_time; long long dump_end_time; };

    struct BBox { int xmin, ymin, xmax, ymax; float score; int class_id; };

    struct PipelineStateMem {
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

        // 【新增】严格对齐 C++ 端的新变量
        int retry_count;
        int max_retry_count;

        long long current_dump_start_time;
        long long truck_load_start_time;
        long long truck_load_end_time;
        long long last_dump_end_time;
        long long last_action_time;

        bool is_truck_active;

        cv::Rect last_dumping_box;
        cv::Rect current_truck_box;
        cv::Rect last_dumping_bucket_box;

        cv::Rect ui_bucket_box;
        cv::Rect ui_truck_box;

        std::vector<BBox> ui_all_detections;

        int stable_frames_remaining;
        bool is_statting;
        int stat_frames_remaining;
        std::vector<float> ratio_buffer;
        float last_avg_ratio;

        std::vector<BucketEvent> pending_bucket_events;
        std::vector<TruckEvent> pending_truck_events;
    };

    PipelineStateMem* state = (PipelineStateMem*)get_pipeline_state(handle);
    if (!state) return;

    jclass resultClass = env->FindClass("com/rosenshine/hhd/Excavator/ExcavatorResult");
    jobject resultObj = env->NewObject(resultClass, env->GetMethodID(resultClass, "<init>", "()V"));

    env->SetIntField(resultObj, env->GetFieldID(resultClass, "currentShovelCount", "I"), state->current_truck_buckets);
    env->SetBooleanField(resultObj, env->GetFieldID(resultClass, "isLoading", "Z"), state->dumping_active);
    env->SetBooleanField(resultObj, env->GetFieldID(resultClass, "isComplete", "Z"), !state->pending_truck_events.empty());
    env->SetBooleanField(resultObj, env->GetFieldID(resultClass, "isStartLoading", "Z"), state->current_truck_buckets >= 1);

    int bucketType = state->bucket_full ? 1 : (state->ui_bucket_box.area() > 0 ? 0 : -1);
    env->SetIntField(resultObj, env->GetFieldID(resultClass, "bucketType", "I"), bucketType);

    jstring jTicketId = env->NewStringUTF(state->ticket_id.c_str());
    env->SetObjectField(resultObj, env->GetFieldID(resultClass, "ticketId", "Ljava/lang/String;"), jTicketId);
    env->DeleteLocalRef(jTicketId);

    jclass rectClass = env->FindClass("android/graphics/Rect");
    jfieldID leftField = env->GetFieldID(rectClass, "left", "I");
    jfieldID topField = env->GetFieldID(rectClass, "top", "I");
    jfieldID rightField = env->GetFieldID(rectClass, "right", "I");
    jfieldID bottomField = env->GetFieldID(rectClass, "bottom", "I");

    jobject bucketRect = env->GetObjectField(resultObj, env->GetFieldID(resultClass, "bucketPosition", "Landroid/graphics/Rect;"));
    env->SetIntField(bucketRect, leftField, state->ui_bucket_box.x);
    env->SetIntField(bucketRect, topField, state->ui_bucket_box.y);
    env->SetIntField(bucketRect, rightField, state->ui_bucket_box.x + state->ui_bucket_box.width);
    env->SetIntField(bucketRect, bottomField, state->ui_bucket_box.y + state->ui_bucket_box.height);

    jobject truckRect = env->GetObjectField(resultObj, env->GetFieldID(resultClass, "truckPosition", "Landroid/graphics/Rect;"));
    env->SetIntField(truckRect, leftField, state->ui_truck_box.x);
    env->SetIntField(truckRect, topField, state->ui_truck_box.y);
    env->SetIntField(truckRect, rightField, state->ui_truck_box.x + state->ui_truck_box.width);
    env->SetIntField(truckRect, bottomField, state->ui_truck_box.y + state->ui_truck_box.height);

    // ================== 全量检测框转码传给 Java ==================
    if (!state->ui_all_detections.empty()) {
        jintArray jAllDetections = env->NewIntArray(state->ui_all_detections.size() * 5);
        if (jAllDetections != nullptr) {
            std::vector<jint> boxData;
            boxData.reserve(state->ui_all_detections.size() * 5);
            for (const auto& box : state->ui_all_detections) {
                boxData.push_back(box.xmin);
                boxData.push_back(box.ymin);
                boxData.push_back(box.xmax);
                boxData.push_back(box.ymax);
                boxData.push_back(box.class_id);
            }
            env->SetIntArrayRegion(jAllDetections, 0, boxData.size(), boxData.data());
            env->SetObjectField(resultObj, env->GetFieldID(resultClass, "allDetections", "[I"), jAllDetections);
            env->DeleteLocalRef(jAllDetections);
        }
    }

    jclass cbClass = env->GetObjectClass(callback);
    env->CallVoidMethod(callback, env->GetMethodID(cbClass, "onResult", "(Lcom/rosenshine/hhd/Excavator/ExcavatorResult;)V"), resultObj);

    jmethodID onBucketMethod = env->GetMethodID(cbClass, "onBucketLoaded", "(Ljava/lang/String;IIJJF)V");
    for (const auto& ev : state->pending_bucket_events) {
        jstring jTicket = env->NewStringUTF(ev.ticket_id.c_str());
        env->CallVoidMethod(callback, onBucketMethod, jTicket, ev.total_truck_count, ev.current_bucket_count, (jlong)ev.dump_start_time, (jlong)ev.dump_end_time, (jfloat)ev.last_mineral_ratio);
        env->DeleteLocalRef(jTicket);
    }

    jmethodID onTruckMethod = env->GetMethodID(cbClass, "onTruckCompleted", "(Ljava/lang/String;IIJJI)V");
    for (const auto& ev : state->pending_truck_events) {
        jstring jTicket = env->NewStringUTF(ev.ticket_id.c_str());
        env->CallVoidMethod(callback, onTruckMethod, jTicket, ev.total_truck_count, ev.total_bucket_count, (jlong)ev.load_start_time, (jlong)ev.load_end_time, (jint)ev.completed_type);
        env->DeleteLocalRef(jTicket);
    }

    clear_pipeline_events(handle);

    env->DeleteLocalRef(resultClass);
    env->DeleteLocalRef(resultObj);
    env->DeleteLocalRef(rectClass);
    env->DeleteLocalRef(bucketRect);
    env->DeleteLocalRef(truckRect);
}

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_updateConfigNative(JNIEnv *env, jclass clazz, jlong handlePtr, jfloat confThresh, jfloat iouThresh) {
    if (handlePtr != 0) update_pipeline_config(reinterpret_cast<void*>(handlePtr), confThresh, iouThresh);
}

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_releaseNative(JNIEnv *env, jclass clazz, jlong handlePtr) {
    if (handlePtr != 0) {
        release_pipeline(reinterpret_cast<void*>(handlePtr));
        env->SetStaticLongField(clazz, env->GetStaticFieldID(clazz, "nativeHandle", "J"), 0);
    }
}

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_restoreStateNative(JNIEnv *env, jclass clazz, jlong handlePtr, jstring ticketId, jint bucketCount, jfloat lastMineralRatio) {
    if (handlePtr == 0) return;
    const char* t_id = env->GetStringUTFChars(ticketId, 0);
    restore_pipeline_state(reinterpret_cast<void*>(handlePtr), t_id, bucketCount, lastMineralRatio);
    env->ReleaseStringUTFChars(ticketId, t_id);
}

JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_setTimeoutNative(JNIEnv *env, jclass clazz, jlong handlePtr, jlong ms) {
    if (handlePtr != 0) set_pipeline_timeout(reinterpret_cast<void*>(handlePtr), ms);
}

} // extern "C"
