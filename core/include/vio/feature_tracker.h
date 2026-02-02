#pragma once

#include "vio/types.h"

#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>

namespace vio {

class FeatureTracker {
public:
    struct Params {
        int max_features = 200;
        double quality_level = 0.01;
        double min_distance = 20.0;
        cv::Size win_size{21, 21};
        int max_pyramid_level = 3;
        int max_track_length = 20;
    };

    explicit FeatureTracker(const Params& params, const CameraIntrinsics& camera);

    // Process a new grayscale image frame.
    // Returns features currently being tracked (with all their observations).
    std::vector<TrackedFeature> process_frame(const cv::Mat& image, Timestamp timestamp);

    // Get features that were lost in the last frame and have 3+ observations.
    // These are candidates for EKF update.
    const std::vector<TrackedFeature>& lost_features() const { return lost_features_; }

private:
    void detect_new_features(const cv::Mat& image);
    void track_features(const cv::Mat& curr_image);
    void remove_outliers_ransac();
    Eigen::Vector2d undistort_point(const cv::Point2f& pt);
    std::vector<TrackedFeature> get_active_tracks();

    Params params_;
    CameraIntrinsics camera_;

    cv::Mat prev_image_;
    std::vector<cv::Point2f> prev_points_;
    std::vector<int> feature_ids_;
    int next_feature_id_ = 0;

    std::unordered_map<int, TrackedFeature> active_tracks_;
    std::vector<TrackedFeature> lost_features_;
};

} // namespace vio
