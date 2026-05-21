#include "excavator_tracker.h"

// 构造函数：初始化所有变量
ExcavatorTracker::ExcavatorTracker() {
    state_change_threshold = 25; // 连续 25 帧才确认状态转换
    reset();
}

ExcavatorTracker::~ExcavatorTracker() {
}

void ExcavatorTracker::reset() {
    total_truck_count = 0;
    total_bucket_count = 0;
    truck_full = false;
    bucket_full = false;
    truck_state_change_count = 0;
}

void ExcavatorTracker::get_counts(int& out_trucks, int& out_buckets) const {
    out_trucks = total_truck_count;
    out_buckets = total_bucket_count;
}

// 检查横坐标是否重叠
bool ExcavatorTracker::check_horizontal_overlap(const DetectObject& box1, const DetectObject& box2) const {
    float x1_min = box1.x1, x1_max = box1.x2;
    float x2_min = box2.x1, x2_max = box2.x2;
    // 如果没有不重叠的情况，那就是重叠了
    return !(x1_max < x2_min || x2_max < x1_min);
}

// 核心状态机更新逻辑
void ExcavatorTracker::update_state_machine(const std::vector<DetectObject>& objects) {
    if (objects.empty()) {
        return;
    }

    std::vector<DetectObject> truck_boxes;
    std::vector<DetectObject> bucket_boxes;

    for (const auto& obj : objects) {
        if (obj.class_id == 2 || obj.class_id == 3) {
            truck_boxes.push_back(obj);
        } else if (obj.class_id == 0 || obj.class_id == 1) {
            bucket_boxes.push_back(obj);
        }
    }

    if (!bucket_boxes.empty()) {
        // 找出置信度最高的 bucket (利用 C++ 的 std::max_element)
        auto best_bucket_it = std::max_element(bucket_boxes.begin(), bucket_boxes.end(),
            [](const DetectObject& a, const DetectObject& b) {
                return a.conf < b.conf;
            });

        const DetectObject& best_bucket = *best_bucket_it;

        if (best_bucket.class_id == 1) { // bucket-full
            if (!bucket_full) {
                bool has_overlap = false;
                for (const auto& truck : truck_boxes) {
                    if (check_horizontal_overlap(best_bucket, truck)) {
                        has_overlap = true;
                        break;
                    }
                }
                // 挖斗与矿车无重叠时，允许变为 full（离开矿车去挖矿）
                if (!has_overlap) {
                    bucket_full = true;
                }
            }
        } else if (best_bucket.class_id == 0) { // bucket-empty
            if (bucket_full) {
                bool has_overlap = false;
                for (const auto& truck : truck_boxes) {
                    if (check_horizontal_overlap(best_bucket, truck)) {
                        has_overlap = true;
                        break;
                    }
                }

                if (has_overlap) {
                    // 有重叠：完成计数，重置状态
                    bucket_full = false;
                    total_bucket_count++;
                } else {
                    // 无重叠：只重置状态，不计数
                    bucket_full = false;
                }
            }
        }
    }

    for (const auto& truck : truck_boxes) {
        if (truck.class_id == 3) { // truck-full
            if (!truck_full) {
                truck_state_change_count++;
                if (truck_state_change_count >= state_change_threshold) {
                    truck_state_change_count = 0;
                    truck_full = true;
                }
            } else {
                truck_state_change_count = 0;
            }
        } else if (truck.class_id == 2) { // truck-empty
            if (truck_full) {
                truck_state_change_count++;
                if (truck_state_change_count >= state_change_threshold) {
                    truck_state_change_count = 0;
                    truck_full = false;
                    total_truck_count++; // 完成一辆车的装载计数
                }
            } else {
                truck_state_change_count = 0;
            }
        }
    }
}
