#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace vio {

using Timestamp = uint64_t;

struct ImuMeasurement {
    Timestamp timestamp;
    Eigen::Vector3d angular_velocity;   // rad/s
    Eigen::Vector3d linear_acceleration; // m/s^2
};

struct ImageData {
    Timestamp timestamp;
    std::string filepath;
};

struct GroundTruthPose {
    Timestamp timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    Eigen::Vector3d velocity;
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d accel_bias;
};

// EKF state: nominal state + 15x15 error-state covariance
struct State {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Timestamp timestamp = 0;

    Eigen::Matrix<double, 15, 15> covariance =
        Eigen::Matrix<double, 15, 15>::Identity() * 1e-3;
};

enum StateIndex {
    POS = 0,
    VEL = 3,
    ORI = 6,
    BG  = 9,
    BA  = 12
};

struct TrackedFeature {
    int id = -1;
    std::vector<Eigen::Vector2d> observations; // normalized camera coords
    std::vector<Timestamp> timestamps;
    Eigen::Vector3d position_3d = Eigen::Vector3d::Zero();
    bool is_triangulated = false;
};

struct ImuNoiseParams {
    double gyro_noise_density    = 1.6968e-4;  // rad/s/sqrt(Hz)
    double accel_noise_density   = 2.0000e-3;  // m/s^2/sqrt(Hz)
    double gyro_random_walk      = 1.9393e-5;  // rad/s^2/sqrt(Hz)
    double accel_random_walk     = 3.0000e-3;  // m/s^3/sqrt(Hz)
};

struct CameraIntrinsics {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<double> distortion;
    int width = 0, height = 0;
    Eigen::Matrix4d T_cam_imu = Eigen::Matrix4d::Identity();
};

} // namespace vio
