package com.sany.excavator;

/**
 * Native bridge to libexcavator_algo.so on RK3568 Android.
 *
 * Lifecycle:
 *   1. ExcavatorNative.init("/sdcard/excavator/best.rknn")
 *   2. Loop: processFrame(cameraBytes, width, height, rotation)
 *   3. ExcavatorNative.release()
 */
public class ExcavatorNative {

    static {
        try {
            System.loadLibrary("excavator_algo");
        } catch (UnsatisfiedLinkError e) {
            // Fallback: explicit path
            System.load("/data/local/tmp/libexcavator_algo.so");
        }
    }

    /** Load the .rknn model and initialize NPU. */
    public static native boolean init(String modelPath);

    /** Release NPU resources. */
    public static native void release();

    /**
     * Process one camera frame through the NPU + counting state machine.
     * @param frameData  RGB or NV21 raw bytes
     * @param width      frame width in pixels
     * @param height     frame height in pixels
     * @param rotation   camera rotation (0, 90, 180, 270)
     * @return JSON string: {"trucks":N,"buckets":M,"detections":[...]}
     */
    public static native String processFrame(byte[] frameData, int width, int height, int rotation);

    /** Get current cumulative counts. */
    public static native int[] getCounts();

    /** Reset all counters. */
    public static native void resetCounts();
}
