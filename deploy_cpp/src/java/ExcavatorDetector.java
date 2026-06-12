package com.rosenshine.hhd.Excavator;

import android.content.res.AssetManager;
import android.os.Handler;
import android.os.Looper;

public class ExcavatorDetector {
    // 静态保存 C++ 层的引擎实例句柄
    private static long nativeHandle = 0;

    static {
        System.loadLibrary("excavator_jni");
    }

    public interface ExcavatorInitCallback {
        void onSuccess();
        void onFailure(String errorMsg);
    }

    // ================= 业务状态与定时器变量 =================
    private static int baseBucketCount = 0;
    private static String currentActiveTicket = "";
    private static int lastReportedBuckets = 0;

    // 【核心新增】：标记当前车辆会话是否已被结算（超时自动结算 或 手动强制结算）
    private static boolean isCurrentTruckSettled = false;

    // 【核心新增】：缓存甲方的回调引用，以便外部手动触发时能够呼叫
    private static ExcavatorCallback registeredClientCallback = null;

    // Android 主线程定时器
    private static final Handler timerHandler = new Handler(Looper.getMainLooper());
    private static Runnable timeoutTask = null;

    // 【改为非 final】：允许外层动态配置超时时间，默认 60000 毫秒（1分钟）
    private static long timeoutMs = 60 * 1000;

    // ================= JNI Native 层真实接口 =================
    public static native void initFromFile(String yoloPath, String siamesePath, ExcavatorInitCallback callback);
    public static native void initFromAsset(AssetManager assetManager, String yoloFileName, String siameseFileName, ExcavatorInitCallback callback);
    public static native void initFromByteArray(byte[] yoloData, byte[] siameseData, ExcavatorInitCallback callback);

    private static native void detectNative(long handle, byte[] yuvData, int width, int height, ExcavatorCallback callback);
    private static native void updateConfigNative(long handle, float confThresh, float iouThresh, float siameseThresh);
    private static native void releaseNative(long handle);

    private static native void restoreStateNative(long handle, String ticketId, int bucketCount, float[] feature);

    // ========================================================
    // 核心暴露接口区：提供给外部 Java/Android 业务层调用的控制台
    // ========================================================

    /**
     * 接口 A：动态设置装车超时阈值（单位：毫秒）
     * 允许甲方在App设置界面动态调整超时结算时间
     */
    public static void setTimeout(long ms) {
        timeoutMs = ms;
    }

    /**
     * 接口 B：获取当前设置的超时时间
     */
    public static long getTimeout() {
        return timeoutMs;
    }

    /**
     * 接口 C：手动强制结算当前卡车
     * 场景：卡车装满直接开走，挖机原地歇工，司机不想等1分钟自动超时，在屏幕上点击“手动结算”按钮
     */
    public static void forceCompleteCurrentTruck() {
        timerHandler.post(new Runnable() {
            @Override
            public void run() {
                if (registeredClientCallback != null && !currentActiveTicket.isEmpty() && !isCurrentTruckSettled) {
                    // 1. 移除倒计时任务
                    if (timeoutTask != null) {
                        timerHandler.removeCallbacks(timeoutTask);
                    }
                    // 2. 立即触发甲方的写库回调
                    registeredClientCallback.onTruckCompleted(currentActiveTicket, lastReportedBuckets);
                    // 3. 锁定状态，防止重复结算
                    isCurrentTruckSettled = true;
                }
            }
        });
    }

    /**
     * 接口 D：查询当前车辆是否已经处于结算状态
     */
    public static boolean isCurrentTruckSettled() {
        return isCurrentTruckSettled;
    }


    // ================= 给甲方调用的公开包装接口 =================

    /**
     * 动态更新底层算法参数（置信度等）
     */
    public static void updateConfig(float confThresh, float iouThresh, float siameseThresh) {
        if (nativeHandle != 0) {
            updateConfigNative(nativeHandle, confThresh, iouThresh, siameseThresh);
        }
    }

    /**
     * 执行挖机动作识别
     */
    public static void detect(byte[] yuvData, int width, int height, final ExcavatorCallback clientCallback) {
        if (nativeHandle == 0 || clientCallback == null) {
            if (clientCallback != null) clientCallback.onResult(null);
            return;
        }

        // 持续刷新外部回调引用
        registeredClientCallback = clientCallback;

        // 数据拦截器
        ExcavatorCallback internalInterceptor = new ExcavatorCallback() {
            @Override
            public void onResult(ExcavatorResult rawResult) {
                if (rawResult == null) {
                    clientCallback.onResult(null);
                    return;
                }

                String incomingTicket = rawResult.getTicketId();

                // 1. 场景一：底层判定换车（新车入场首铲）
                if (rawResult.isComplete()) {
                    // 如果前一辆车由于某些原因还没有被结算，立刻强制结算它
                    if (!currentActiveTicket.isEmpty() && !isCurrentTruckSettled && !currentActiveTicket.equals(incomingTicket)) {
                        if (timeoutTask != null) timerHandler.removeCallbacks(timeoutTask);
                        clientCallback.onTruckCompleted(currentActiveTicket, lastReportedBuckets);
                    }

                    // 初始化全新的车辆会话
                    currentActiveTicket = incomingTicket;
                    baseBucketCount = Math.max(0, rawResult.getCurrentShovelCount() - 1);
                    isCurrentTruckSettled = false;
                }

                // 2. 场景二：午休/故障断档重连（已被结算的旧车，镜头转回来又装了一铲）
                if (incomingTicket != null && incomingTicket.equals(currentActiveTicket) && isCurrentTruckSettled) {
                    // 孪生网络判定还是原来的车！此时激活会话，取消结算锁定
                    isCurrentTruckSettled = false;
                    // 注意：这里绝不重置 baseBucketCount，这样数值就能“接着之前的斗数继续数”
                }

                // 3. 计算当前卡车的实时相对斗数
                int actualBuckets = 0;
                if (incomingTicket != null && incomingTicket.equals(currentActiveTicket)) {
                    actualBuckets = Math.max(0, rawResult.getCurrentShovelCount() - baseBucketCount);
                }
                rawResult.setCurrentShovelCount(actualBuckets);
                lastReportedBuckets = actualBuckets;

                // 4. 定时器管理：只要车辆处于未结算的装车状态，就疯狂刷新倒计时
                if (incomingTicket != null && !incomingTicket.isEmpty() && !isCurrentTruckSettled) {
                    if (timeoutTask != null) {
                        timerHandler.removeCallbacks(timeoutTask);
                    }

                    final String ticketForTimer = incomingTicket;
                    final int bucketsForTimer = actualBuckets;

                    timeoutTask = new Runnable() {
                        @Override
                        public void run() {
                            // 炸弹爆炸：到达指定超时时间没有新装矿，通知甲方写库
                            clientCallback.onTruckCompleted(ticketForTimer, bucketsForTimer);
                            isCurrentTruckSettled = true; // 标记为已结算，进入等待续装或换车状态
                        }
                    };
                    // 使用暴露出来的动态超时变量
                    timerHandler.postDelayed(timeoutTask, timeoutMs);
                }

                // 5. 将精修好的数据发给甲方 UI 渲染
                clientCallback.onResult(rawResult);
            }

            @Override
            public void onTruckCompleted(String ticketId, int totalBuckets) {
                // 拦截器内部不处理，交由上面的业务逻辑和定时器触发
            }
        };

        // 把拦截器实例传给底层 C++
        detectNative(nativeHandle, yuvData, width, height, internalInterceptor);
    }

    /**
     * 释放模型资源
     */
    public static void release() {
        timerHandler.removeCallbacksAndMessages(null);
        timeoutTask = null;
        currentActiveTicket = "";
        registeredClientCallback = null;
        isCurrentTruckSettled = false;

        if (nativeHandle != 0) {
            releaseNative(nativeHandle);
        }
    }

    /**
     * 【重要新增】：断电/重启 状态恢复接口
     * 在调用 init 成功后立即调用此方法，将数据库中最后一次未结算的车辆特征灌入。
     * @param ticketId 上次断电前最后正在装载的票号
     * @param lastBuckets 上次断电前这辆车已经装载的真实斗数
     * @param feature 上次断电前保存的 256维特征向量 (从 getTruckFeature() 获得)
     */
    public static void restoreState(String ticketId, int lastBuckets, float[] feature) {
        if (nativeHandle != 0 && feature != null && feature.length > 0) {
            // 1. 将数据灌入 C++ 底层感知器
            restoreStateNative(nativeHandle, ticketId, lastBuckets, feature);

            // 2. 同步 Java 层的业务状态机
            currentActiveTicket = ticketId;
            // 【核心数学逻辑】：因为底层 C++ 的绝对斗数被我们强行恢复成了 lastBuckets。
            // 为了让后续相减 (实际斗数 = 绝对斗数 - 基数) 的公式依然成立并顺延，
            // 此时 Java 层的基数必须设为 0。
            baseBucketCount = 0;
            isCurrentTruckSettled = false;
            lastReportedBuckets = lastBuckets;
        }
    }
}