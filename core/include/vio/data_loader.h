#pragma once

#include "vio/types.h"
#include <string>
#include <vector>

namespace vio {

class DataLoader {
public:
    explicit DataLoader(const std::string& dataset_path);

    bool load();

    const std::vector<ImuMeasurement>& imu_data() const { return imu_data_; }
    const std::vector<ImageData>& cam0_images() const { return cam0_images_; }
    const std::vector<GroundTruthPose>& ground_truth() const { return ground_truth_; }
    const CameraIntrinsics& camera_intrinsics() const { return camera_; }

private:
    bool parse_imu_csv(const std::string& filepath);
    bool parse_image_csv(const std::string& filepath, std::vector<ImageData>& out,
                         const std::string& image_dir);
    bool parse_groundtruth_csv(const std::string& filepath);
    bool parse_sensor_yaml(const std::string& filepath);

    std::string dataset_path_;
    std::vector<ImuMeasurement> imu_data_;
    std::vector<ImageData> cam0_images_;
    std::vector<GroundTruthPose> ground_truth_;
    CameraIntrinsics camera_;
};

} // namespace vio
