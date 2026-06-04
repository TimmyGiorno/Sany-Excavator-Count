package com.rosenshine.hhd.Excavator;

import android.graphics.Rect;

public class ExcavatorResult {
    private boolean isStartLoading;       // 是否开始装车
    private Rect bucketPosition;          // 挖机铲斗位置（屏幕坐标矩形）
    private int bucketType;               // 0=挖斗(空)，1=翻斗(满)，-1=无铲斗
    private boolean isLoading;            // 是否正在装车(发生卸料动作)
    private Rect truckPosition;           // 自卸车车斗位置（屏幕坐标矩形）
    private int currentShovelCount;       // 当前第几铲（从1开始）
    private boolean isComplete;           // 是否装车完成(换车)

    // JNI C++ 层专用的构造函数（必须存在，且需初始化 Rect 防止空指针）
    public ExcavatorResult() {
        this.bucketPosition = new Rect();
        this.truckPosition = new Rect();
        this.bucketType = -1;
    }

    // ================= Getter & Setter =================
    public boolean isStartLoading() { return isStartLoading; }
    public void setStartLoading(boolean startLoading) { isStartLoading = startLoading; }

    public Rect getBucketPosition() { return bucketPosition; }
    public void setBucketPosition(Rect bucketPosition) { this.bucketPosition = bucketPosition; }

    public int getBucketType() { return bucketType; }
    public void setBucketType(int bucketType) { this.bucketType = bucketType; }

    public boolean isLoading() { return isLoading; }
    public void setLoading(boolean loading) { isLoading = loading; }

    public Rect getTruckPosition() { return truckPosition; }
    public void setTruckPosition(Rect truckPosition) { this.truckPosition = truckPosition; }

    public int getCurrentShovelCount() { return currentShovelCount; }
    public void setCurrentShovelCount(int currentShovelCount) { this.currentShovelCount = currentShovelCount; }

    public boolean isComplete() { return isComplete; }
    public void setComplete(boolean complete) { isComplete = complete; }
}