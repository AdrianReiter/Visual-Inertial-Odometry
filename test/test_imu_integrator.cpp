#include "vio/imu_integrator.h"

#include <cassert>
#include <cmath>
#include <iostream>

static void test_stationary() {
    // Stationary IMU: only gravity in z-up world frame.
    // Body frame aligned with world: accelerometer reads [0, 0, +9.81] (cancels gravity).
    // Position and velocity should remain near zero.
    vio::ImuIntegrator integrator;
    vio::State state;
    state.timestamp = 0;

    std::vector<vio::ImuMeasurement> measurements;
    for (int i = 0; i < 200; ++i) {
        vio::ImuMeasurement m;
        m.timestamp = static_cast<vio::Timestamp>(i) * 5000000ULL; // 5ms intervals = 200Hz
        m.angular_velocity = Eigen::Vector3d::Zero();
        m.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81); // Cancels gravity
        measurements.push_back(m);
    }

    vio::Timestamp t_end = 199 * 5000000ULL;
    integrator.propagate(state, measurements, 0, t_end);

    // Position should stay near zero (1 second of stationary data)
    std::cout << "Stationary test - position: " << state.position.transpose() << "\n";
    std::cout << "Stationary test - velocity: " << state.velocity.transpose() << "\n";
    assert(state.position.norm() < 0.01 && "Stationary position drifted too much");
    assert(state.velocity.norm() < 0.01 && "Stationary velocity drifted too much");

    // Covariance should have grown from initial value
    double initial_trace = 15 * 1e-3;
    double final_trace = state.covariance.trace();
    assert(final_trace > initial_trace && "Covariance should grow during propagation");

    std::cout << "test_stationary: PASSED\n";
}

static void test_constant_acceleration() {
    // Constant acceleration of 1 m/s^2 in x direction (world frame).
    // Body aligned with world. Accelerometer reads [1.0, 0, 9.81].
    // After 1 second: v_x = 1.0 m/s, p_x = 0.5 m.
    vio::ImuIntegrator integrator;
    vio::State state;
    state.timestamp = 0;

    std::vector<vio::ImuMeasurement> measurements;
    for (int i = 0; i < 200; ++i) {
        vio::ImuMeasurement m;
        m.timestamp = static_cast<vio::Timestamp>(i) * 5000000ULL;
        m.angular_velocity = Eigen::Vector3d::Zero();
        m.linear_acceleration = Eigen::Vector3d(1.0, 0.0, 9.81);
        measurements.push_back(m);
    }

    vio::Timestamp t_end = 199 * 5000000ULL;
    integrator.propagate(state, measurements, 0, t_end);

    double total_time = t_end * 1e-9;
    double expected_vx = 1.0 * total_time;
    double expected_px = 0.5 * 1.0 * total_time * total_time;

    std::cout << "Const accel test - position: " << state.position.transpose()
              << " (expected x=" << expected_px << ")\n";
    std::cout << "Const accel test - velocity: " << state.velocity.transpose()
              << " (expected vx=" << expected_vx << ")\n";

    assert(std::abs(state.velocity.x() - expected_vx) < 0.05 &&
           "Velocity x should be ~1.0 m/s");
    assert(std::abs(state.position.x() - expected_px) < 0.05 &&
           "Position x should be ~0.5 m");
    assert(std::abs(state.velocity.y()) < 0.01 && "Velocity y should be ~0");
    assert(std::abs(state.velocity.z()) < 0.01 && "Velocity z should be ~0");

    std::cout << "test_constant_acceleration: PASSED\n";
}

static void test_constant_rotation() {
    // Constant rotation about z-axis at 1 rad/s for ~1 second.
    // After 1 second, should have rotated ~1 radian about z.
    vio::ImuIntegrator integrator;
    vio::State state;
    state.timestamp = 0;

    std::vector<vio::ImuMeasurement> measurements;
    for (int i = 0; i < 200; ++i) {
        vio::ImuMeasurement m;
        m.timestamp = static_cast<vio::Timestamp>(i) * 5000000ULL;
        m.angular_velocity = Eigen::Vector3d(0.0, 0.0, 1.0); // 1 rad/s about z
        m.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        measurements.push_back(m);
    }

    vio::Timestamp t_end = 199 * 5000000ULL;
    integrator.propagate(state, measurements, 0, t_end);

    // Extract yaw angle from quaternion
    Eigen::Vector3d euler = state.orientation.toRotationMatrix().canonicalEulerAngles(2, 1, 0);
    double yaw = euler[0];
    double expected_yaw = t_end * 1e-9; // ~0.995 radians

    std::cout << "Rotation test - yaw: " << yaw << " (expected ~" << expected_yaw << ")\n";
    assert(std::abs(yaw - expected_yaw) < 0.05 &&
           "Yaw should be ~1 radian after 1 second");

    std::cout << "test_constant_rotation: PASSED\n";
}

int main() {
    test_stationary();
    test_constant_acceleration();
    test_constant_rotation();
    std::cout << "All IMU integrator tests passed.\n";
    return 0;
}
