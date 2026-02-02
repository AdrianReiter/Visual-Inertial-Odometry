#pragma once

#include "vio/types.h"
#include <vector>

namespace vio {

class ImuIntegrator {
public:
    explicit ImuIntegrator(const ImuNoiseParams& noise_params = ImuNoiseParams());

    // Propagate state from t_start to t_end using IMU measurements in that interval.
    // Modifies state in-place (position, velocity, orientation, covariance).
    // If Phi_out is non-null, accumulates the full state transition matrix across
    // all IMU steps (product of per-step F matrices).
    void propagate(State& state,
                   const std::vector<ImuMeasurement>& measurements,
                   Timestamp t_start,
                   Timestamp t_end,
                   Eigen::Matrix<double, 15, 15>* Phi_out = nullptr);

private:
    void integrate_single(State& state,
                          const Eigen::Vector3d& omega,
                          const Eigen::Vector3d& accel,
                          double dt);

    Eigen::Matrix<double, 15, 15> compute_F(const State& state,
                                             const Eigen::Vector3d& accel_corrected,
                                             const Eigen::Vector3d& omega_corrected,
                                             double dt);
    Eigen::Matrix<double, 15, 12> compute_G(const State& state, double dt);
    Eigen::Matrix<double, 12, 12> compute_Q(double dt);

    static Eigen::Matrix3d skew(const Eigen::Vector3d& v);

    ImuNoiseParams noise_;
    Eigen::Vector3d gravity_{0.0, 0.0, -9.81};
};

} // namespace vio
