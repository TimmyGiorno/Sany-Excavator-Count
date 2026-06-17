package com.rosenshine.hhd.Excavator;

public interface ExcavatorCallback {
    void onResult(ExcavatorResult result);

    // 装满一铲的事件
    void onBucketLoaded(String ticketId, int totalTruckCount, int currentBucket, long dumpStartTime, long dumpEndTime, float lastMineralRatio);

    // 卡车开走或超时（业务完结）的事件 (0为正常开走，1为超时强制结束)
    void onTruckCompleted(String ticketId, int totalTruckCount, int totalBuckets, long loadStartTime, long loadEndTime, int completedType);
}