package com.rosenshine.hhd.Excavator;

import android.graphics.Rect;

public class ExcavatorResult {
    private String ticketId;
    private boolean isStartLoading;
    private Rect bucketPosition;
    private int bucketType;
    private boolean isLoading;
    private Rect truckPosition;
    private int currentShovelCount;
    private boolean isComplete;

    public ExcavatorResult() {
        this.bucketPosition = new Rect();
        this.truckPosition = new Rect();
        this.bucketType = -1;
        this.ticketId = "";
    }

    public String getTicketId() { return ticketId; }
    public void setTicketId(String ticketId) { this.ticketId = ticketId; }

    public boolean isStartLoading() { return isStartLoading; }
    public void setStartLoading(boolean startLoading) { isStartLoading = startLoading; }

    public Rect getBucketPosition() { return bucketPosition; }
    public void setBucketPosition(Rect bucketPosition) { this.bucketPosition = bucketPosition; }

    public int getBucketType() { return bucketType; }
    public void setBucketType(int bucketType) { this.bucketType = bucketType; }

    public boolean isLoading() { return isLoading; }
    public void setLoading(boolean loading) { this.isLoading = loading; }

    public Rect getTruckPosition() { return truckPosition; }
    public void setTruckPosition(Rect truckPosition) { this.truckPosition = truckPosition; }

    public int getCurrentShovelCount() { return currentShovelCount; }
    public void setCurrentShovelCount(int currentShovelCount) { this.currentShovelCount = currentShovelCount; }

    public boolean isComplete() { return isComplete; }
    public void setComplete(boolean complete) { this.isComplete = complete; }
}