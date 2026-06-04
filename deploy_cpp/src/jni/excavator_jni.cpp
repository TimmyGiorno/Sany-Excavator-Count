#include <jni.h>
#include <string>
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "excavator_pipeline.h"

// 让 C++ 代码把报错和信息打印到 Android 的 Logcat 日志里
#define LOG_TAG "ExcavatorJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// 辅助函数：触发 Java 的回调
void triggerCallback(JNIEnv *env, jobject callback, bool success, const char* errorMsg = "") {
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

// 从文件路径加载模型
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromFile(JNIEnv *env, jclass clazz, jstring yoloPath, jstring siamesePath, jobject callback) {
    const char *yolo_path_c = env->GetStringUTFChars(yoloPath, 0);
    const char *siamese_path_c = env->GetStringUTFChars(siamesePath, 0);

    int yolo_size = 0, siamese_size = 0;
    unsigned char* yolo_buf = loadFileToMemory(yolo_path_c, &yolo_size);
    unsigned char* siamese_buf = loadFileToMemory(siamese_path_c, &siamese_size);

    if (!yolo_buf || !siamese_buf) {
        triggerCallback(env, callback, false, "Failed to read model files from path");
    } else {
        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size, siamese_buf, siamese_size);
        
        // C++ 层的对象指针存入 Java 的 nativeHandle 变量中
        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));
        
        triggerCallback(env, callback, true);
    }

    if (yolo_buf) free(yolo_buf);
    if (siamese_buf) free(siamese_buf);
    env->ReleaseStringUTFChars(yoloPath, yolo_path_c);
    env->ReleaseStringUTFChars(siamesePath, siamese_path_c);
}

// 从 Asset 加载模型
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromAsset(JNIEnv *env, jclass clazz, jobject assetManager, jstring yoloFileName, jstring siameseFileName, jobject callback) {
    // 获取 Android 原生的 AssetManager
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    
    const char *yolo_name_c = env->GetStringUTFChars(yoloFileName, 0);
    const char *siamese_name_c = env->GetStringUTFChars(siameseFileName, 0);

    AAsset* yolo_asset = AAssetManager_open(mgr, yolo_name_c, AASSET_MODE_BUFFER);
    AAsset* siamese_asset = AAssetManager_open(mgr, siamese_name_c, AASSET_MODE_BUFFER);

    if (!yolo_asset || !siamese_asset) {
        triggerCallback(env, callback, false, "Failed to open model files from assets");
    } else {
        const void* yolo_buf = AAsset_getBuffer(yolo_asset);
        off_t yolo_size = AAsset_getLength(yolo_asset);
        
        const void* siamese_buf = AAsset_getBuffer(siamese_asset);
        off_t siamese_size = AAsset_getLength(siamese_asset);

        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size, siamese_buf, siamese_size);

        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));

        triggerCallback(env, callback, true);
    }

    if (yolo_asset) AAsset_close(yolo_asset);
    if (siamese_asset) AAsset_close(siamese_asset);
    env->ReleaseStringUTFChars(yoloFileName, yolo_name_c);
    env->ReleaseStringUTFChars(siameseFileName, siamese_name_c);
}

// 从字节数组加载模型
JNIEXPORT void JNICALL
Java_com_rosenshine_hhd_Excavator_ExcavatorDetector_initFromByteArray(JNIEnv *env, jclass clazz, jbyteArray yoloData, jbyteArray siameseData, jobject callback) {
    // 将 Java 的 byte[] 转换为 C++ 的指针
    jsize yolo_size = env->GetArrayLength(yoloData);
    jbyte* yolo_buf = env->GetByteArrayElements(yoloData, NULL);
    
    jsize siamese_size = env->GetArrayLength(siameseData);
    jbyte* siamese_buf = env->GetByteArrayElements(siameseData, NULL);

    if (yolo_size == 0 || siamese_size == 0) {
        triggerCallback(env, callback, false, "Byte array is empty");
    } else {
        void* handle = init_pipeline_from_memory(yolo_buf, yolo_size, siamese_buf, siamese_size);

        jfieldID handleField = env->GetStaticFieldID(clazz, "nativeHandle", "J");
        env->SetStaticLongField(clazz, handleField, reinterpret_cast<jlong>(handle));

        triggerCallback(env, callback, true);
    }

    // 释放 JNI 数组锁定
    env->ReleaseByteArrayElements(yoloData, yolo_buf, JNI_ABORT);
    env->ReleaseByteArrayElements(siameseData, siamese_buf, JNI_ABORT);
}

} // extern "C"