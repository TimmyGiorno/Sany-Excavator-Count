#include <jni.h>
#include <string>
#include <cstring>
#include <vector>
#include <mutex>
#include <cstdio>

#include "../include/inference_engine.h"
#include "../include/excavator_tracker.h"

/* Factory for RKNN backend -- defined in backends/rknn_backend.cpp */
IBackend* create_rknn_backend();

/* Globals */
static InferenceEngine   g_engine;
static ExcavatorTracker  g_tracker;
static std::mutex        g_mutex;
static bool              g_initialized = false;

static int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_sany_excavator_ExcavatorNative_init(JNIEnv* env, jclass clazz, jstring modelPath) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_initialized) return JNI_TRUE;

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    if (!path) return JNI_FALSE;

    IBackend* backend = create_rknn_backend();
    bool ok = g_engine.load_model(backend, path);
    env->ReleaseStringUTFChars(modelPath, path);

    if (ok) {
        g_tracker.reset();
        g_initialized = true;
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_sany_excavator_ExcavatorNative_release(JNIEnv* env, jclass clazz) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_initialized = false;
}
JNIEXPORT jstring JNICALL
Java_com_sany_excavator_ExcavatorNative_processFrame(
        JNIEnv* env, jclass clazz,
        jbyteArray frameData, jint width, jint height, jint rotation) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_initialized) {
        return env->NewStringUTF("{\"error\":\"not_initialized\"}");
    }

    jbyte* data = env->GetByteArrayElements(frameData, nullptr);
    if (!data) {
        return env->NewStringUTF("{\"error\":\"null_data\"}");
    }

    int w = clamp_int(width,  1, 4096);
    int h = clamp_int(height, 1, 4096);

    std::vector<DetectBox> boxes = g_engine.detect(
        reinterpret_cast<const uint8_t*>(data), w, h);

    env->ReleaseByteArrayElements(frameData, data, JNI_ABORT);

    std::vector<DetectObject> objects;
    for (const auto& b : boxes) {
        DetectObject obj;
        obj.class_id = b.class_id;
        obj.track_id = 0;
        obj.conf     = b.confidence;
        obj.x1 = b.x1; obj.y1 = b.y1;
        obj.x2 = b.x2; obj.y2 = b.y2;
        objects.push_back(obj);
    }

    g_tracker.update_state_machine(objects);
    int trucks, buckets;
    g_tracker.get_counts(trucks, buckets);
    char buf[8192];
    int off = snprintf(buf, sizeof(buf),
        "{\"trucks\":%d,\"buckets\":%d,\"detections\":[",
        trucks, buckets);

    bool first = true;
    for (const auto& b : boxes) {
        const char* names[] = {"bucket-empty","bucket-full","truck-empty","truck-full"};
        const char* cls_name = (b.class_id >= 0 && b.class_id < 4)
            ? names[b.class_id] : "unknown";

        int rem = (int)sizeof(buf) - off;
        if (rem < 200) break;

        if (!first) buf[off++] = ',';
        first = false;
        off += snprintf(buf + off, rem,
            "{\"x1\":%.1f,\"y1\":%.1f,\"x2\":%.1f,\"y2\":%.1f,"
            "\"conf\":%.2f,\"cls\":\"%s\"}",
            b.x1, b.y1, b.x2, b.y2, b.confidence, cls_name);
    }

    int rem = (int)sizeof(buf) - off;
    if (rem > 3) {
        off += snprintf(buf + off, rem, "]}");
    }

    return env->NewStringUTF(buf);
}

JNIEXPORT jintArray JNICALL
Java_com_sany_excavator_ExcavatorNative_getCounts(JNIEnv* env, jclass clazz) {
    std::lock_guard<std::mutex> lock(g_mutex);
    int trucks, buckets;
    g_tracker.get_counts(trucks, buckets);
    jintArray result = env->NewIntArray(2);
    jint vals[2] = { trucks, buckets };
    env->SetIntArrayRegion(result, 0, 2, vals);
    return result;
}

JNIEXPORT void JNICALL
Java_com_sany_excavator_ExcavatorNative_resetCounts(JNIEnv* env, jclass clazz) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_tracker.reset();
}

} /* extern "C" */
