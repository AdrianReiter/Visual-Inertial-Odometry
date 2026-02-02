#pragma once

#include "vio/imu_integrator.h"
#include "vio/types.h"
#include <vector>

namespace vio {

class EkfEstimator {
public:
    struct Params {
        ImuNoiseParams imu_noise;
        double feature_noise_std = 1.5; // pixels (before normalization)
    };

    struct AugmentedPose {
        Timestamp timestamp;
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation;
        int state_index; // starting column/row index in full_cov_
    };

    explicit EkfEstimator(const Params& params, const CameraIntrinsics& camera);

    void initialize(const State& initial_state);

    // IMU prediction step
    void predict(const std::vector<ImuMeasurement>& imu_measurements,
                 Timestamp t_target);

    // Visual measurement update using lost/mature features (MSCKF)
    void update(const std::vector<TrackedFeature>& features);

    // Clone current camera pose into the augmented state
    void store_camera_pose(Timestamp timestamp);

    const State& state() const { return state_; }

private:
    // Clone current IMU pose into augmented state, expanding covariance
    void clone_camera_pose(Timestamp timestamp);

    // Remove an augmented pose by index, shrinking covariance
    void marginalize_pose(int idx);

    bool triangulate_feature(TrackedFeature& feature);

    // Compute MSCKF residuals for one feature across all its augmented pose observations.
    // Returns false if insufficient observations or degenerate geometry.
    // H_out: (2M-3) x state_dim_, r_out: (2M-3) x 1 after null-space projection
    bool compute_feature_residuals(const TrackedFeature& feature,
                                   Eigen::MatrixXd& H_out,
                                   Eigen::VectorXd& r_out);

    // Project a world point into normalized camera coords given a body pose
    Eigen::Vector2d project(const Eigen::Vector3d& point_world,
                            const Eigen::Vector3d& p_body,
                            const Eigen::Quaterniond& q_body);

    // Apply Kalman update with Joseph form on full augmented state
    void apply_update(const Eigen::MatrixXd& H, const Eigen::VectorXd& r,
                      const Eigen::MatrixXd& R);

    // Inject error state into nominal state (IMU + augmented poses)
    void inject_error_state(const Eigen::VectorXd& delta_x);

    // Find augmented pose closest to a timestamp
    const AugmentedPose* find_augmented_pose(Timestamp timestamp) const;

    // Find index of augmented pose closest to a timestamp (-1 if not found)
    int find_augmented_pose_index(Timestamp timestamp) const;

    static Eigen::Matrix3d skew(const Eigen::Vector3d& v);

    State state_;
    Params params_;
    CameraIntrinsics camera_;
    ImuIntegrator imu_integrator_;

    // MSCKF augmented state
    Eigen::MatrixXd full_cov_;                    // (15 + 6*N) x (15 + 6*N)
    std::vector<AugmentedPose> augmented_poses_;
    int state_dim_ = 15;                           // current total state dimension

    static constexpr int kImuStateDim = 15;
    static constexpr int kPoseDim = 6;             // 3 position + 3 orientation
    static constexpr int kMaxAugmentedPoses = 20;
};

} // namespace vio
