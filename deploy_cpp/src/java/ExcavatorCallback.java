package com.rosenshine.hhd.Excavator;

public interface ExcavatorCallback {
    void onResult(ExcavatorResult result);

    // 装满一铲的事件
    void onBucketLoaded(String ticketId, int totalTruckCount, int currentBucket, long dumpStartTime, long dumpEndTime);

    // 卡车开走（业务完结）的事件
    void onTruckCompleted(String ticketId, int totalTruckCount, int totalBuckets, long loadStartTime, long loadEndTime);

    // 车辆作业中途停工超时的预警事件 (一辆车只会触发一次，不会打断业务)
    void onTimeout(String ticketId);
}