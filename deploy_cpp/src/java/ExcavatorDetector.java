package com.rosenshine.hhd.Excavator;

import android.content.res.AssetManager;

public class ExcavatorDetector {
    // 静态保存 C++ 层的引擎实例句柄
    private static long nativeHandle = 0;

    static {
        // 加载刚才用 CMake 编译出来的动态库
        System.loadLibrary("excavator_jni");
    }

    public interface ExcavatorInitCallback {
        void onSuccess();
        void onFailure(String errorMsg);
    }

    // ================= JNI Native 层真实接口 (私有/按需暴露) =================

    // 初始化方法（直接暴露给甲方使用）
    public static native void initFromFile(String yoloPath, String siamesePath, ExcavatorInitCallback callback);
    public static native void initFromAsset(AssetManager assetManager, String yoloFileName, String siameseFileName, ExcavatorInitCallback callback);
    public static native void initFromByteArray(byte[] yoloData, byte[] siameseData, ExcavatorInitCallback callback);

    // 带 handle 的底层检测与释放接口（私有，对甲方隐藏）
    private static native void detectNative(long handle, byte[] yuvData, int width, int height, ExcavatorCallback callback);
    private static native void releaseNative(long handle);


    // ================= 给甲方调用的公开包装接口 =================

    /**
     * 执行挖机动作识别（异步）
     * @param yuvData   YUV420 NV21/NV12 格式图像数据
     * @param width     图像宽度
     * @param height    图像高度
     * @param callback  识别结果回调
     */
    public static void detect(byte[] yuvData, int width, int height, ExcavatorCallback callback) {
        if (nativeHandle != 0) {
            // 自动补上 handle 发给 C++ 底层
            detectNative(nativeHandle, yuvData, width, height, callback);
        } else {
            if (callback != null) {
                // 引擎未初始化，抛出空结果防崩溃
                callback.onResult(null);
            }
        }
    }

    /**
     * 释放模型资源
     */
    public static void release() {
        if (nativeHandle != 0) {
            releaseNative(nativeHandle);
            // nativeHandle 会在 JNI 底层被重置为 0，防止野指针
        }
    }
}