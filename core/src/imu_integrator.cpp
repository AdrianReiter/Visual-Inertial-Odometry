#include "vio/imu_integrator.h"

#include <algorithm>

namespace vio {

ImuIntegrator::ImuIntegrator(const ImuNoiseParams& noise_params)
    : noise_(noise_params) {}

void ImuIntegrator::propagate(State& state,
                               const std::vector<ImuMeasurement>& measurements,
                               Timestamp t_start,
                               Timestamp t_end,
                               Eigen::Matrix<double, 15, 15>* Phi_out) {
    if (Phi_out) {
        *Phi_out = Eigen::Matrix<double, 15, 15>::Identity();
    }

    // Find IMU measurements in [t_start, t_end]
    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& m = measurements[i];
        if (m.timestamp < t_start) continue;
        if (m.timestamp > t_end) break;

        // Determine dt: time from previous integration point to this measurement
        Timestamp t_prev = (i > 0 && measurements[i - 1].timestamp >= t_start)
                               ? measurements[i - 1].timestamp
                               : t_start;
        // For the last measurement, integrate up to t_end
        Timestamp t_curr = m.timestamp;
        if (i + 1 < measurements.size() && measurements[i + 1].timestamp <= t_end) {
            // Use midpoint between consecutive measurements
            t_curr = m.timestamp;
        } else {
            t_curr = std::min(m.timestamp, t_end);
        }

        double dt = (t_curr - t_prev) * 1e-9;
        if (dt <= 0.0) continue;

        Eigen::Vector3d omega = m.angular_velocity - state.gyro_bias;
        Eigen::Vector3d accel = m.linear_acceleration - state.accel_bias;

        // Update covariance: P = F*P*F^T + G*Q*G^T
        auto F = compute_F(state, accel, omega, dt);
        auto G = compute_G(state, dt);
        auto Q = compute_Q(dt);
        state.covariance = F * state.covariance * F.transpose() +
                           G * Q * G.transpose();

        // Accumulate state transition matrix
        if (Phi_out) {
            *Phi_out = F * (*Phi_out);
        }
        // Enforce symmetry
        state.covariance = 0.5 * (state.covariance + state.covariance.transpose());

        // Integrate nominal state
        integrate_single(state, omega, accel, dt);
        state.timestamp = t_curr;
    }

    state.timestamp = t_end;
}

void ImuIntegrator::integrate_single(State& state,
                                      const Eigen::Vector3d& omega,
                                      const Eigen::Vector3d& accel,
                                      double dt) {
    Eigen::Matrix3d R = state.orientation.toRotationMatrix();
    Eigen::Vector3d accel_world = R * accel + gravity_;

    // Position update (midpoint)
    state.position += state.velocity * dt + 0.5 * accel_world * dt * dt;

    // Velocity update
    state.velocity += accel_world * dt;

    // Orientation update: q = q * delta_q(omega * dt)
    double angle = omega.norm() * dt;
    if (angle > 1e-12) {
        Eigen::Vector3d axis = omega.normalized();
        Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, axis));
        state.orientation = (state.orientation * dq).normalized();
    }
}

Eigen::Matrix<double, 15, 15> ImuIntegrator::compute_F(
    const State& state,
    const Eigen::Vector3d& accel_corrected,
    const Eigen::Vector3d& omega_corrected,
    double dt) {
    Eigen::Matrix3d R = state.orientation.toRotationMatrix();

    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();

    // dp/dv
    F.block<3, 3>(POS, VEL) = Eigen::Matrix3d::Identity() * dt;

    // dv/dtheta = -R * [a]x * dt
    F.block<3, 3>(VEL, ORI) = -R * skew(accel_corrected) * dt;

    // dv/dba = -R * dt
    F.block<3, 3>(VEL, BA) = -R * dt;

    // dtheta/dtheta = I - [omega]x * dt
    F.block<3, 3>(ORI, ORI) = Eigen::Matrix3d::Identity() - skew(omega_corrected) * dt;

    // dtheta/dbg = -I * dt
    F.block<3, 3>(ORI, BG) = -Eigen::Matrix3d::Identity() * dt;

    return F;
}

Eigen::Matrix<double, 15, 12> ImuIntegrator::compute_G(const State& state,
                                                         double dt) {
    Eigen::Matrix3d R = state.orientation.toRotationMatrix();
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();

    // Noise order: [gyro_noise(3), accel_noise(3), gyro_walk(3), accel_walk(3)]
    // dv/d(accel_noise) = -R * dt
    G.block<3, 3>(VEL, 3) = -R * dt;

    // dtheta/d(gyro_noise) = -I * dt
    G.block<3, 3>(ORI, 0) = -Eigen::Matrix3d::Identity() * dt;

    // dbg/d(gyro_walk) = I * dt
    G.block<3, 3>(BG, 6) = Eigen::Matrix3d::Identity() * dt;

    // dba/d(accel_walk) = I * dt
    G.block<3, 3>(BA, 9) = Eigen::Matrix3d::Identity() * dt;

    return G;
}

Eigen::Matrix<double, 12, 12> ImuIntegrator::compute_Q(double dt) {
    Eigen::Matrix<double, 12, 12> Q = Eigen::Matrix<double, 12, 12>::Zero();

    double ng2 = noise_.gyro_noise_density * noise_.gyro_noise_density;
    double na2 = noise_.accel_noise_density * noise_.accel_noise_density;
    double ngw2 = noise_.gyro_random_walk * noise_.gyro_random_walk;
    double naw2 = noise_.accel_random_walk * noise_.accel_random_walk;

    // Continuous-time noise spectral densities, divided by dt because
    // G already includes dt, and the discrete Q = Qc / dt
    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * (ng2 / dt);
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * (na2 / dt);
    Q.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * (ngw2 / dt);
    Q.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * (naw2 / dt);

    return Q;
}

Eigen::Matrix3d ImuIntegrator::skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
         v.z(),      0, -v.x(),
        -v.y(),  v.x(),      0;
    return m;
}

} // namespace vio
