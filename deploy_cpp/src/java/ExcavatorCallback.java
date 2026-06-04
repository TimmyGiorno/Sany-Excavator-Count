package com.rosenshine.hhd.Excavator;

public interface ExcavatorCallback {
    /**
     * 识别结果回调（每帧一次）
     * @param result 识别结果对象
     */
    void onResult(ExcavatorResult result);
}