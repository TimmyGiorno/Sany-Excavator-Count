package com.rosenshine.hhd.Excavator;

public interface ExcavatorCallback {
    /**
     * 识别结果回调（每帧一次，用于UI画框和实时数据显示）
     * @param result 识别结果对象
     */
    void onResult(ExcavatorResult result);

    /**
     * 【新增】装车彻底完成回调（用于触发写入数据库）
     * 触发时机：1. 距离最后一次识别到该车超过 1 分钟  或 2. 提前识别到了下一辆新车
     * @param ticketId   完成的票号
     * @param totalBuckets 这辆车的最终总斗数
     */
    void onTruckCompleted(String ticketId, int totalBuckets);
}