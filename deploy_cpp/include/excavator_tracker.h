#ifndef EXCAVATOR_TRACKER_H
#define EXCAVATOR_TRACKER_H

#include <vector>
#include <algorithm>

struct DetectObject {
    int class_id; // 0: bucket-empty, 1: bucket-full, 2: truck-empty, 3: truck-full
    int track_id;
    float conf;
    float x1, y1, x2, y2;
};

class ExcavatorTracker {
public:
    ExcavatorTracker();
    ~ExcavatorTracker();

    void reset();
    void get_counts(int& out_trucks, int& out_buckets) const;
    void update_state_machine(const std::vector<DetectObject>& objects);

private:
    bool check_horizontal_overlap(const DetectObject& box1, const DetectObject& box2) const;

    int total_truck_count;
    int total_bucket_count;
    bool truck_full;
    bool bucket_full;
    int truck_state_change_count;
    int state_change_threshold;
};

#endif //EXCAVATOR_TRACKER_H
