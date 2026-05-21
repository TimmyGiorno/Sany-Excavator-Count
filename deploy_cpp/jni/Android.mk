LOCAL_PATH := $(call my-dir)

# ---------- librknnrt (prebuilt stub for linking) ----------
include $(CLEAR_VARS)
LOCAL_MODULE := rknnrt
LOCAL_SRC_FILES := ../3rdparty/rknn/librknnrt.so
include $(PREBUILT_SHARED_LIBRARY)

# ---------- excavator_algo (.so) ----------
include $(CLEAR_VARS)
LOCAL_MODULE := excavator_algo

LOCAL_SRC_FILES := \
    excavator_jni.cpp \
    ../src/inference_engine.cpp \
    ../src/backends/rknn_backend.cpp \
    ../src/excavator_tracker.cpp

LOCAL_C_INCLUDES := \
    $(LOCAL_PATH) \
    $(LOCAL_PATH)/../include

LOCAL_CFLAGS := -std=c++17 -Wall -O2 -fPIC -DUSE_RKNN

LOCAL_LDLIBS := -llog -landroid

LOCAL_SHARED_LIBRARIES := rknnrt

include $(BUILD_SHARED_LIBRARY)
