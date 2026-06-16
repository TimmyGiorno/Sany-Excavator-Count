package com.rosenshine.hhd.Excavator;

import android.content.res.AssetManager;

public class ExcavatorDetector {
    private static long nativeHandle = 0;

    static {
        System.loadLibrary("excavator_jni");
    }

    public interface ExcavatorInitCallback {
        void onSuccess();
        void onFailure(String errorMsg);
    }

    // ================= JNI Native 接口 =================
    public static native void initFromFile(String yoloPath, ExcavatorInitCallback callback);
    public static native void initFromAsset(AssetManager assetManager, String yoloFileName, ExcavatorInitCallback callback);
    public static native void initFromByteArray(byte[] yoloData, ExcavatorInitCallback callback);

    private static native void detectNative(long handle, byte[] yuvData, int width, int height, ExcavatorCallback callback);
    private static native void updateConfigNative(long handle, float confThresh, float iouThresh);
    private static native void releaseNative(long handle);

    private static native void restoreStateNative(long handle, String ticketId, int bucketCount, float lastMineralRatio);
    private static native void setTimeoutNative(long handle, long ms);

    // ================= 开放给客户端的接口 =================

    public static void setTimeout(long ms) {
        if (nativeHandle != 0) setTimeoutNative(nativeHandle, ms);
    }

    public static void updateConfig(float confThresh, float iouThresh) {
        if (nativeHandle != 0) updateConfigNative(nativeHandle, confThresh, iouThresh);
    }

    public static void detect(byte[] yuvData, int width, int height, ExcavatorCallback clientCallback) {
        if (nativeHandle == 0 || clientCallback == null) {
            if (clientCallback != null) clientCallback.onResult(null);
            return;
        }
        detectNative(nativeHandle, yuvData, width, height, clientCallback);
    }

    public static void restoreState(String ticketId, int lastBuckets, float lastMineralRatio) {
        if (nativeHandle != 0) {
            restoreStateNative(nativeHandle, ticketId, lastBuckets, lastMineralRatio);
        }
    }

    public static void release() {
        if (nativeHandle != 0) {
            releaseNative(nativeHandle);
            nativeHandle = 0;
        }
    }
}