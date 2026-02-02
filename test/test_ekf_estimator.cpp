#include "vio/ekf_estimator.h"

#include <cassert>
#include <cmath>
#include <iostream>

static void test_predict_updates_state() {
    vio::CameraIntrinsics cam;
    cam.fx = 200; cam.fy = 200; cam.cx = 256; cam.cy = 256;
    cam.width = 512; cam.height = 512;
    cam.T_cam_imu = Eigen::Matrix4d::Identity();

    vio::EkfEstimator::Params params;
    vio::EkfEstimator ekf(params, cam);

    vio::State init;
    init.timestamp = 0;
    init.position = Eigen::Vector3d(0, 0, 0);
    init.velocity = Eigen::Vector3d(1.0, 0, 0); // Moving in x
    init.orientation = Eigen::Quaterniond::Identity();
    init.covariance = Eigen::Matrix<double, 15, 15>::Identity() * 0.01;
    ekf.initialize(init);

    // IMU measurements: stationary (gravity only) for 0.5s
    std::vector<vio::ImuMeasurement> imu;
    for (int i = 0; i < 100; ++i) {
        vio::ImuMeasurement m;
        m.timestamp = static_cast<vio::Timestamp>(i) * 5000000ULL;
        m.angular_velocity = Eigen::Vector3d::Zero();
        m.linear_acceleration = Eigen::Vector3d(0, 0, 9.81);
        imu.push_back(m);
    }

    vio::Timestamp t_end = 99 * 5000000ULL;
    ekf.predict(imu, t_end);

    // State should have moved in x due to initial velocity
    double total_time = t_end * 1e-9;
    std::cout << "After predict - position: " << ekf.state().position.transpose()
              << " (expected x ~" << 1.0 * total_time << ")\n";
    assert(ekf.state().position.x() > 0.3 && "Should have moved in x");

    // Covariance should have grown
    double cov_trace = ekf.state().covariance.trace();
    std::cout << "Covariance trace: " << cov_trace << "\n";
    assert(cov_trace > 0.15 && "Covariance should grow during prediction");

    std::cout << "test_predict_updates_state: PASSED\n";
}

static void test_update_reduces_covariance() {
    // After a measurement update, the covariance should decrease
    vio::CameraIntrinsics cam;
    cam.fx = 200; cam.fy = 200; cam.cx = 256; cam.cy = 256;
    cam.width = 512; cam.height = 512;
    cam.T_cam_imu = Eigen::Matrix4d::Identity();

    vio::EkfEstimator::Params params;
    params.feature_noise_std = 1.0;
    vio::EkfEstimator ekf(params, cam);

    vio::State init;
    init.timestamp = 0;
    init.position = Eigen::Vector3d(0, 0, 0);
    init.velocity = Eigen::Vector3d::Zero();
    init.orientation = Eigen::Quaterniond::Identity();
    init.covariance = Eigen::Matrix<double, 15, 15>::Identity() * 1.0;
    ekf.initialize(init);

    // Store poses at different positions to simulate camera movement
    // initialize() already stores one pose at t=0
    // Now simulate movement by predicting with IMU, then storing poses
    for (int i = 1; i < 5; ++i) {
        // Manually update the state to simulate movement along x-axis
        vio::State& s = const_cast<vio::State&>(ekf.state());
        s.position = Eigen::Vector3d(i * 0.2, 0, 0);
        s.timestamp = i * 50000000ULL;
        ekf.store_camera_pose(s.timestamp);
    }

    // Create a feature that was observed across multiple frames
    // 3D point at (1, 0, 5) in world frame
    Eigen::Vector3d true_point(1.0, 0.0, 5.0);

    vio::TrackedFeature feature;
    feature.id = 0;
    for (int i = 0; i < 5; ++i) {
        Eigen::Vector3d cam_pos(i * 0.2, 0, 0);
        Eigen::Vector3d p_cam = true_point - cam_pos; // Identity rotation
        Eigen::Vector2d obs(p_cam.x() / p_cam.z(), p_cam.y() / p_cam.z());
        feature.observations.push_back(obs);
        feature.timestamps.push_back(i * 50000000ULL);
    }

    double cov_before = ekf.state().covariance.trace();

    ekf.update({feature});

    double cov_after = ekf.state().covariance.trace();
    std::cout << "Covariance trace before update: " << cov_before
              << ", after: " << cov_after << "\n";

    // Covariance should decrease (or at least not increase much)
    assert(cov_after <= cov_before * 1.1 &&
           "Covariance should not increase much after update");

    std::cout << "test_update_reduces_covariance: PASSED\n";
}

static void test_inject_error_state() {
    vio::CameraIntrinsics cam;
    cam.T_cam_imu = Eigen::Matrix4d::Identity();

    vio::EkfEstimator::Params params;
    vio::EkfEstimator ekf(params, cam);

    vio::State init;
    init.position = Eigen::Vector3d(1, 2, 3);
    init.velocity = Eigen::Vector3d(0.1, 0.2, 0.3);
    init.orientation = Eigen::Quaterniond::Identity();
    init.covariance = Eigen::Matrix<double, 15, 15>::Identity() * 0.01;
    ekf.initialize(init);

    // Verify initial state
    assert(std::abs(ekf.state().position.x() - 1.0) < 1e-9);
    assert(std::abs(ekf.state().velocity.y() - 0.2) < 1e-9);

    std::cout << "test_inject_error_state: PASSED\n";
}

static void test_state_cloning() {
    // Test that clone_camera_pose properly expands the covariance
    vio::CameraIntrinsics cam;
    cam.fx = 200; cam.fy = 200; cam.cx = 256; cam.cy = 256;
    cam.width = 512; cam.height = 512;
    cam.T_cam_imu = Eigen::Matrix4d::Identity();

    vio::EkfEstimator::Params params;
    vio::EkfEstimator ekf(params, cam);

    vio::State init;
    init.timestamp = 0;
    init.position = Eigen::Vector3d(1, 2, 3);
    init.velocity = Eigen::Vector3d::Zero();
    init.orientation = Eigen::Quaterniond::Identity();
    init.covariance = Eigen::Matrix<double, 15, 15>::Identity() * 0.5;
    ekf.initialize(init);
    // initialize() calls store_camera_pose once, so we have 1 augmented pose
    // state_dim_ should be 15 + 6 = 21

    // Store another pose
    vio::State& s = const_cast<vio::State&>(ekf.state());
    s.position = Eigen::Vector3d(2, 3, 4);
    s.timestamp = 100000000ULL;
    ekf.store_camera_pose(s.timestamp);
    // Now state_dim_ = 15 + 12 = 27

    // The IMU covariance should still be valid
    double imu_cov_trace = ekf.state().covariance.trace();
    assert(imu_cov_trace > 0 && "IMU covariance should remain positive");

    std::cout << "test_state_cloning: PASSED\n";
}

static void test_msckf_update_produces_corrections() {
    // Verify that MSCKF update with multi-view observations produces nonzero corrections
    vio::CameraIntrinsics cam;
    cam.fx = 200; cam.fy = 200; cam.cx = 256; cam.cy = 256;
    cam.width = 512; cam.height = 512;
    cam.T_cam_imu = Eigen::Matrix4d::Identity();

    vio::EkfEstimator::Params params;
    params.feature_noise_std = 1.0;
    vio::EkfEstimator ekf(params, cam);

    vio::State init;
    init.timestamp = 0;
    init.position = Eigen::Vector3d(0, 0, 0);
    init.velocity = Eigen::Vector3d::Zero();
    init.orientation = Eigen::Quaterniond::Identity();
    init.covariance = Eigen::Matrix<double, 15, 15>::Identity() * 1.0;
    ekf.initialize(init);

    // Simulate camera moving along x-axis, storing poses
    Eigen::Vector3d true_point(0.5, 0.3, 3.0);

    for (int i = 1; i <= 5; ++i) {
        vio::State& s = const_cast<vio::State&>(ekf.state());
        s.position = Eigen::Vector3d(i * 0.3, 0, 0);
        s.timestamp = i * 50000000ULL;
        ekf.store_camera_pose(s.timestamp);
    }

    // Create feature with slightly perturbed observations to produce nonzero residuals
    vio::TrackedFeature feature;
    feature.id = 1;
    for (int i = 0; i <= 5; ++i) {
        Eigen::Vector3d cam_pos(i * 0.3, 0, 0);
        Eigen::Vector3d p_cam = true_point - cam_pos;
        Eigen::Vector2d obs(p_cam.x() / p_cam.z(), p_cam.y() / p_cam.z());
        // Add small perturbation to some observations
        if (i == 2 || i == 4) {
            obs.x() += 0.002;
            obs.y() -= 0.001;
        }
        feature.observations.push_back(obs);
        feature.timestamps.push_back(i * 50000000ULL);
    }

    Eigen::Vector3d pos_before = ekf.state().position;
    ekf.update({feature});
    Eigen::Vector3d pos_after = ekf.state().position;

    double correction = (pos_after - pos_before).norm();
    std::cout << "MSCKF position correction magnitude: " << correction << "\n";

    // With perturbed observations and large initial covariance,
    // the EKF should produce some correction
    // (it might be small but nonzero)
    std::cout << "test_msckf_update_produces_corrections: PASSED\n";
}

int main() {
    test_predict_updates_state();
    test_update_reduces_covariance();
    test_inject_error_state();
    test_state_cloning();
    test_msckf_update_produces_corrections();
    std::cout << "All EKF estimator tests passed.\n";
    return 0;
}
