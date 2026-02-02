#include "vio/ekf_estimator.h"
#include "vio/triangulation.h"

#include <Eigen/QR>
#include <iostream>

namespace vio {

EkfEstimator::EkfEstimator(const Params& params, const CameraIntrinsics& camera)
    : params_(params), camera_(camera), imu_integrator_(params.imu_noise) {}

void EkfEstimator::initialize(const State& initial_state) {
    state_ = initial_state;
    augmented_poses_.clear();
    state_dim_ = kImuStateDim;
    full_cov_ = state_.covariance;
    store_camera_pose(state_.timestamp);
}

void EkfEstimator::predict(const std::vector<ImuMeasurement>& imu_measurements,
                            Timestamp t_target) {
    Eigen::Matrix<double, 15, 15> Phi;
    imu_integrator_.propagate(state_, imu_measurements, state_.timestamp, t_target, &Phi);
    state_.timestamp = t_target;

    // Copy updated 15x15 IMU covariance into full_cov_ top-left block
    full_cov_.block<kImuStateDim, kImuStateDim>(0, 0) = state_.covariance;

    // Update cross-correlations between IMU state and augmented poses
    int aug_dim = state_dim_ - kImuStateDim;
    if (aug_dim > 0) {
        full_cov_.block(0, kImuStateDim, kImuStateDim, aug_dim) =
            Phi * full_cov_.block(0, kImuStateDim, kImuStateDim, aug_dim);
        full_cov_.block(kImuStateDim, 0, aug_dim, kImuStateDim) =
            full_cov_.block(0, kImuStateDim, kImuStateDim, aug_dim).transpose();
    }
}

void EkfEstimator::store_camera_pose(Timestamp timestamp) {
    clone_camera_pose(timestamp);
}

void EkfEstimator::clone_camera_pose(Timestamp timestamp) {
    // Enforce max window size
    while (static_cast<int>(augmented_poses_.size()) >= kMaxAugmentedPoses) {
        marginalize_pose(0);
    }

    AugmentedPose pose;
    pose.timestamp = timestamp;
    pose.position = state_.position;
    pose.orientation = state_.orientation;
    pose.state_index = state_dim_;

    // Jacobian mapping IMU error state to cloned pose error
    // Clone pose = [position, orientation] from IMU state
    // J_clone is 6 x 15: maps 15D IMU error to 6D pose error
    Eigen::Matrix<double, 6, Eigen::Dynamic> J_clone =
        Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, state_dim_);
    // Position block: dp_clone/dp_imu = I
    J_clone.block<3, 3>(0, StateIndex::POS) = Eigen::Matrix3d::Identity();
    // Orientation block: dtheta_clone/dtheta_imu = I
    J_clone.block<3, 3>(3, StateIndex::ORI) = Eigen::Matrix3d::Identity();

    // Expand covariance
    int new_dim = state_dim_ + kPoseDim;
    Eigen::MatrixXd new_cov = Eigen::MatrixXd::Zero(new_dim, new_dim);

    // Copy existing covariance
    new_cov.block(0, 0, state_dim_, state_dim_) = full_cov_;

    // New rows: J_clone * P (cross-correlation with all existing states)
    Eigen::MatrixXd P_cross = J_clone * full_cov_;  // 6 x state_dim_
    new_cov.block(state_dim_, 0, kPoseDim, state_dim_) = P_cross;
    new_cov.block(0, state_dim_, state_dim_, kPoseDim) = P_cross.transpose();

    // New diagonal block: J_clone * P * J_clone^T
    new_cov.block(state_dim_, state_dim_, kPoseDim, kPoseDim) =
        J_clone * full_cov_ * J_clone.transpose();

    full_cov_ = new_cov;
    state_dim_ = new_dim;
    augmented_poses_.push_back(pose);

    // Enforce symmetry
    full_cov_ = 0.5 * (full_cov_ + full_cov_.transpose());
}

void EkfEstimator::marginalize_pose(int idx) {
    if (idx < 0 || idx >= static_cast<int>(augmented_poses_.size())) return;

    int si = augmented_poses_[idx].state_index;

    // Remove 6 rows and columns at si from full_cov_
    int new_dim = state_dim_ - kPoseDim;
    Eigen::MatrixXd new_cov = Eigen::MatrixXd::Zero(new_dim, new_dim);

    // Copy blocks around the removed rows/cols
    // Top-left: [0..si) x [0..si)
    if (si > 0) {
        new_cov.block(0, 0, si, si) = full_cov_.block(0, 0, si, si);
    }
    // Top-right: [0..si) x [si+6..state_dim_)
    int after = state_dim_ - si - kPoseDim;
    if (si > 0 && after > 0) {
        new_cov.block(0, si, si, after) =
            full_cov_.block(0, si + kPoseDim, si, after);
    }
    // Bottom-left: [si+6..state_dim_) x [0..si)
    if (after > 0 && si > 0) {
        new_cov.block(si, 0, after, si) =
            full_cov_.block(si + kPoseDim, 0, after, si);
    }
    // Bottom-right: [si+6..state_dim_) x [si+6..state_dim_)
    if (after > 0) {
        new_cov.block(si, si, after, after) =
            full_cov_.block(si + kPoseDim, si + kPoseDim, after, after);
    }

    full_cov_ = new_cov;
    state_dim_ = new_dim;

    augmented_poses_.erase(augmented_poses_.begin() + idx);

    // Update state_index for remaining poses
    for (auto& ap : augmented_poses_) {
        if (ap.state_index > si) {
            ap.state_index -= kPoseDim;
        }
    }
}

void EkfEstimator::update(const std::vector<TrackedFeature>& features) {
    if (features.empty()) return;

    // Collect all valid MSCKF residuals
    std::vector<Eigen::MatrixXd> H_list;
    std::vector<Eigen::VectorXd> r_list;
    int total_rows = 0;

    for (auto feature : features) {
        if (!triangulate_feature(feature)) continue;

        Eigen::MatrixXd H_o;
        Eigen::VectorXd r_o;
        if (compute_feature_residuals(feature, H_o, r_o)) {
            H_list.push_back(H_o);
            r_list.push_back(r_o);
            total_rows += static_cast<int>(r_o.rows());
        }
    }

    if (H_list.empty()) return;

    // Stack all features
    Eigen::MatrixXd H_stacked(total_rows, state_dim_);
    Eigen::VectorXd r_stacked(total_rows);
    H_stacked.setZero();
    r_stacked.setZero();

    int row = 0;
    for (size_t i = 0; i < H_list.size(); ++i) {
        int rows = static_cast<int>(r_list[i].rows());
        H_stacked.block(row, 0, rows, state_dim_) = H_list[i];
        r_stacked.segment(row, rows) = r_list[i];
        row += rows;
    }

    // Measurement noise (after null-space projection, still use same sigma)
    double sigma_n = params_.feature_noise_std / camera_.fx;
    double sigma2 = sigma_n * sigma_n;
    Eigen::MatrixXd R_stacked = Eigen::MatrixXd::Identity(total_rows, total_rows) * sigma2;

    apply_update(H_stacked, r_stacked, R_stacked);
}

bool EkfEstimator::triangulate_feature(TrackedFeature& feature) {
    if (feature.observations.size() < 3) return false;

    std::vector<Eigen::Vector2d> obs;
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> positions;

    Eigen::Matrix3d R_cam_body = camera_.T_cam_imu.block<3, 3>(0, 0);
    Eigen::Vector3d t_cam_body = camera_.T_cam_imu.block<3, 1>(0, 3);

    for (size_t i = 0; i < feature.timestamps.size(); ++i) {
        const AugmentedPose* pose = find_augmented_pose(feature.timestamps[i]);
        if (!pose) continue;

        Eigen::Matrix3d R_body = pose->orientation.toRotationMatrix();
        Eigen::Vector3d p_cam = pose->position + R_body * t_cam_body;
        Eigen::Matrix3d R_cam = R_cam_body * R_body.transpose();

        obs.push_back(feature.observations[i]);
        rotations.push_back(R_cam);
        positions.push_back(p_cam);
    }

    if (obs.size() < 2) return false;

    return Triangulator::triangulate(obs, rotations, positions, feature.position_3d);
}

bool EkfEstimator::compute_feature_residuals(const TrackedFeature& feature,
                                              Eigen::MatrixXd& H_out,
                                              Eigen::VectorXd& r_out) {
    Eigen::Matrix3d R_cam_body = camera_.T_cam_imu.block<3, 3>(0, 0);
    Eigen::Vector3d t_cam_body = camera_.T_cam_imu.block<3, 1>(0, 3);

    // Collect observations that have matching augmented poses
    struct ObsData {
        Eigen::Vector2d obs;
        int pose_idx; // index into augmented_poses_
    };
    std::vector<ObsData> valid_obs;

    for (size_t i = 0; i < feature.timestamps.size(); ++i) {
        int pidx = find_augmented_pose_index(feature.timestamps[i]);
        if (pidx >= 0) {
            valid_obs.push_back({feature.observations[i], pidx});
        }
    }

    int M = static_cast<int>(valid_obs.size());
    if (M < 2) return false;

    // Build per-observation Jacobians and residuals
    // H_x: (2M x state_dim_) Jacobian w.r.t. state
    // H_f: (2M x 3) Jacobian w.r.t. 3D feature point
    // r: (2M x 1) residual
    Eigen::MatrixXd H_x = Eigen::MatrixXd::Zero(2 * M, state_dim_);
    Eigen::MatrixXd H_f = Eigen::MatrixXd::Zero(2 * M, 3);
    Eigen::VectorXd r = Eigen::VectorXd::Zero(2 * M);

    for (int k = 0; k < M; ++k) {
        const auto& od = valid_obs[k];
        const AugmentedPose& ap = augmented_poses_[od.pose_idx];

        Eigen::Matrix3d R_body = ap.orientation.toRotationMatrix();
        Eigen::Vector3d p_cam_world = ap.position + R_body * t_cam_body;
        Eigen::Matrix3d R_cam = R_cam_body * R_body.transpose();

        // Point in camera frame
        Eigen::Vector3d p_cf = R_cam * (feature.position_3d - p_cam_world);

        if (std::abs(p_cf.z()) < 1e-6) return false;

        double z_inv = 1.0 / p_cf.z();
        double z_inv2 = z_inv * z_inv;

        // Jacobian of projection [x/z, y/z] w.r.t. p_cf
        Eigen::Matrix<double, 2, 3> J_proj;
        J_proj << z_inv, 0, -p_cf.x() * z_inv2,
                  0, z_inv, -p_cf.y() * z_inv2;

        // Residual
        Eigen::Vector2d projected(p_cf.x() * z_inv, p_cf.y() * z_inv);
        r.segment<2>(2 * k) = od.obs - projected;

        // --- Jacobian w.r.t. augmented pose position ---
        // dp_cf/dp_body = -R_cam (from p_cf = R_cam * (pw - p_cam_world))
        // dp_cam_world/dp_body = I (position error directly affects p_cam_world)
        Eigen::Matrix3d dp_cf_dp = -R_cam;

        // --- Jacobian w.r.t. augmented pose orientation ---
        // Using the same derivation as the original code
        Eigen::Vector3d p_diff = feature.position_3d - p_cam_world;
        Eigen::Vector3d v = R_body.transpose() * p_diff;
        Eigen::Matrix3d skew_v = skew(v);
        // REMOVED: Eigen::Matrix3d skew_t = skew(t_cam_body);
        // Correct Jacobian: dp_cf / dtheta_body = R_cam_body * [v]_x
        // The lever arm t_cam_body is constant in the body frame and does not contribute to the rotational derivative here.
        Eigen::Matrix3d dp_cf_dtheta = R_cam_body.transpose() * skew_v;        // WAS: Eigen::Matrix3d dp_cf_dtheta = R_cam_body * (skew_v + skew_t)
        // Place into H_x at the augmented pose's state_index
        int si = ap.state_index;
        H_x.block<2, 3>(2 * k, si)     = J_proj * dp_cf_dp;      // position
        H_x.block<2, 3>(2 * k, si + 3) = J_proj * dp_cf_dtheta;  // orientation

        // --- Jacobian w.r.t. 3D feature point ---
        // dp_cf/dp_f = R_cam
        H_f.block<2, 3>(2 * k, 0) = J_proj * R_cam;
    }

    // Null-space projection to eliminate feature point dependency
    // H_f is (2M x 3). We need the left null-space of H_f.
    // QR decomposition: H_f = Q * R, where Q is (2M x 2M), R is (2M x 3)
    // Left null-space = last (2M - 3) columns of Q
    int null_dim = 2 * M - 3;
    if (null_dim <= 0) return false; // Need at least 2 observations (4 rows), giving null_dim=1

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(H_f);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(2 * M, 2 * M);

    // The null-space basis is columns [3..2M) of Q
    Eigen::MatrixXd A = Q.rightCols(null_dim); // (2M x null_dim)

    // Project out the feature point dependency
    H_out = A.transpose() * H_x;  // (null_dim x state_dim_)
    r_out = A.transpose() * r;    // (null_dim x 1)

    // Chi-squared outlier rejection on the projected residual
    double sigma_n = params_.feature_noise_std / camera_.fx;
    double sigma2 = sigma_n * sigma_n;
    double mahal = r_out.squaredNorm() / sigma2;
    // Threshold: chi2 with null_dim degrees of freedom at 95%
    // Approximate: 2*null_dim for large dof
    double threshold = 2.0 * null_dim + 3.0 * std::sqrt(4.0 * null_dim);
    if (mahal > threshold) return false;

    return true;
}

Eigen::Vector2d EkfEstimator::project(const Eigen::Vector3d& point_world,
                                       const Eigen::Vector3d& p_body,
                                       const Eigen::Quaterniond& q_body) {
    Eigen::Matrix3d R_cam_body = camera_.T_cam_imu.block<3, 3>(0, 0);
    Eigen::Vector3d t_cam_body = camera_.T_cam_imu.block<3, 1>(0, 3);

    Eigen::Matrix3d R_body = q_body.toRotationMatrix();
    Eigen::Vector3d p_cam = p_body + R_body * t_cam_body;
    Eigen::Matrix3d R_cam = R_cam_body.transpose() * R_body.transpose();

    Eigen::Vector3d p_cf = R_cam * (point_world - p_cam);

    if (std::abs(p_cf.z()) < 1e-8) {
        return Eigen::Vector2d(1e6, 1e6);
    }

    return Eigen::Vector2d(p_cf.x() / p_cf.z(), p_cf.y() / p_cf.z());
}

void EkfEstimator::apply_update(const Eigen::MatrixXd& H,
                                 const Eigen::VectorXd& r,
                                 const Eigen::MatrixXd& R) {
    // Innovation covariance: S = H * P * H^T + R
    Eigen::MatrixXd S = H * full_cov_ * H.transpose() + R;

    // Kalman gain: K = P * H^T * S^-1
    Eigen::MatrixXd K = full_cov_ * H.transpose() * S.inverse();

    // Error state correction
    Eigen::VectorXd delta_x = K * r;

    // Joseph form covariance update
    Eigen::MatrixXd I_KH =
        Eigen::MatrixXd::Identity(state_dim_, state_dim_) - K * H;
    full_cov_ = I_KH * full_cov_ * I_KH.transpose() + K * R * K.transpose();

    // Enforce symmetry
    full_cov_ = 0.5 * (full_cov_ + full_cov_.transpose());

    // Copy IMU block back to state_.covariance for compatibility
    state_.covariance = full_cov_.block<kImuStateDim, kImuStateDim>(0, 0);

    inject_error_state(delta_x);
}

void EkfEstimator::inject_error_state(const Eigen::VectorXd& delta_x) {
    // IMU state correction
    state_.position += delta_x.segment<3>(StateIndex::POS);
    state_.velocity += delta_x.segment<3>(StateIndex::VEL);

    Eigen::Vector3d dtheta = delta_x.segment<3>(StateIndex::ORI);
    Eigen::Quaterniond dq;
    dq.w() = 1.0;
    dq.x() = 0.5 * dtheta.x();
    dq.y() = 0.5 * dtheta.y();
    dq.z() = 0.5 * dtheta.z();
    state_.orientation = (state_.orientation * dq).normalized();

    state_.gyro_bias += delta_x.segment<3>(StateIndex::BG);
    state_.accel_bias += delta_x.segment<3>(StateIndex::BA);

    // Augmented pose corrections
    for (auto& ap : augmented_poses_) {
        int si = ap.state_index;
        if (si + kPoseDim > delta_x.size()) continue;

        ap.position += delta_x.segment<3>(si);

        Eigen::Vector3d dtheta_pose = delta_x.segment<3>(si + 3);
        Eigen::Quaterniond dq_pose;
        dq_pose.w() = 1.0;
        dq_pose.x() = 0.5 * dtheta_pose.x();
        dq_pose.y() = 0.5 * dtheta_pose.y();
        dq_pose.z() = 0.5 * dtheta_pose.z();
        ap.orientation = (ap.orientation * dq_pose).normalized();
    }
}

const EkfEstimator::AugmentedPose* EkfEstimator::find_augmented_pose(
    Timestamp timestamp) const {
    int idx = find_augmented_pose_index(timestamp);
    if (idx >= 0) return &augmented_poses_[idx];
    return nullptr;
}

int EkfEstimator::find_augmented_pose_index(Timestamp timestamp) const {
    int best_idx = -1;
    uint64_t best_diff = UINT64_MAX;

    for (int i = 0; i < static_cast<int>(augmented_poses_.size()); ++i) {
        uint64_t diff = (augmented_poses_[i].timestamp > timestamp)
                            ? (augmented_poses_[i].timestamp - timestamp)
                            : (timestamp - augmented_poses_[i].timestamp);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }

    // Allow up to 50ms tolerance
    if (best_idx >= 0 && best_diff < 50000000ULL) {
        return best_idx;
    }
    return -1;
}

Eigen::Matrix3d EkfEstimator::skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

} // namespace vio
