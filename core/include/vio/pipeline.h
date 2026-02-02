#pragma once

#include "vio/data_loader.h"
#include "vio/ekf_estimator.h"
#include "vio/feature_tracker.h"
#include "vio/types.h"

#include <functional>
#include <string>
#include <vector>

namespace vio {

class Pipeline {
public:
    struct Config {
        std::string dataset_path;
        bool visualize = true;
    };

    // Callback for each processed frame
    using PoseCallback = std::function<void(
        const State& estimated,
        const GroundTruthPose* ground_truth,
        const std::vector<Eigen::Vector3d>& feature_points,
        const cv::Mat& image)>;

    explicit Pipeline(const Config& config);

    // Run the full pipeline. Calls pose_callback after each frame.
    void run(PoseCallback pose_callback = nullptr);

    const DataLoader& data_loader() const { return data_loader_; }

private:
    void initialize_from_groundtruth();
    const GroundTruthPose* find_nearest_groundtruth(Timestamp t) const;

    // Get IMU measurements in time range [t0, t1]
    std::vector<ImuMeasurement> get_imu_between(Timestamp t0, Timestamp t1) const;

    Config config_;
    DataLoader data_loader_;
    std::unique_ptr<FeatureTracker> feature_tracker_;
    std::unique_ptr<EkfEstimator> ekf_;
};

} // namespace vio
