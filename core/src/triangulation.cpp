#include "vio/triangulation.h"

#include <ceres/ceres.h>
#include <ceres/autodiff_cost_function.h>

namespace vio {

struct ReprojectionCost {
    ReprojectionCost(const Eigen::Vector2d& obs,
                     const Eigen::Matrix3d& R,
                     const Eigen::Vector3d& t)
        : observed(obs), R_cam_world(R), cam_pos(t) {}

    template <typename T>
    bool operator()(const T* const point, T* residuals) const {
        // Transform world point to camera frame: p_cam = R * (p_world - t)
        Eigen::Matrix<T, 3, 1> p_world(point[0], point[1], point[2]);
        Eigen::Matrix<T, 3, 1> t_T = cam_pos.cast<T>();
        Eigen::Matrix<T, 3, 3> R_T = R_cam_world.cast<T>();

        Eigen::Matrix<T, 3, 1> p_cam = R_T * (p_world - t_T);

        // Pinhole projection: [x/z, y/z]
        if (ceres::abs(p_cam(2)) < T(1e-8)) {
            residuals[0] = T(1e6);
            residuals[1] = T(1e6);
            return true;
        }

        T projected_x = p_cam(0) / p_cam(2);
        T projected_y = p_cam(1) / p_cam(2);

        residuals[0] = projected_x - T(observed.x());
        residuals[1] = projected_y - T(observed.y());
        return true;
    }

    Eigen::Vector2d observed;
    Eigen::Matrix3d R_cam_world;
    Eigen::Vector3d cam_pos;
};

bool Triangulator::triangulate(
    const std::vector<Eigen::Vector2d>& observations,
    const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& positions,
    Eigen::Vector3d& point_3d) {

    if (observations.size() < 2) return false;

    // Linear triangulation for initial estimate
    point_3d = triangulate_linear(observations, rotations, positions);

    // Check if point is finite
    if (!point_3d.allFinite()) return false;

    // Refine with Ceres
    if (!refine_ceres(observations, rotations, positions, point_3d)) {
        return false;
    }

    // Check that point is in front of at least one camera
    for (size_t i = 0; i < observations.size(); ++i) {
        Eigen::Vector3d p_cam = rotations[i] * (point_3d - positions[i]);
        if (p_cam.z() > 0.1) return true; // Point is in front of this camera
    }
    return false; // Point behind all cameras
}

Eigen::Vector3d Triangulator::triangulate_linear(
    const std::vector<Eigen::Vector2d>& observations,
    const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& positions) {

    // DLT triangulation: solve A * p = 0
    int n = static_cast<int>(observations.size());
    Eigen::MatrixXd A(2 * n, 4);

    for (int i = 0; i < n; ++i) {
        // Projection matrix P = K * [R | -R*t] but since we use normalized coords, K=I
        // P = [R | -R*t] (3x4)
        Eigen::Matrix<double, 3, 4> P;
        P.block<3, 3>(0, 0) = rotations[i];
        P.block<3, 1>(0, 3) = -rotations[i] * positions[i];

        double u = observations[i].x();
        double v = observations[i].y();

        A.row(2 * i)     = u * P.row(2) - P.row(0);
        A.row(2 * i + 1) = v * P.row(2) - P.row(1);
    }

    // SVD: solution is last column of V
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d sol = svd.matrixV().col(3);

    if (std::abs(sol(3)) < 1e-10) {
        return Eigen::Vector3d(0, 0, 0); // Degenerate
    }

    return sol.head<3>() / sol(3);
}

bool Triangulator::refine_ceres(
    const std::vector<Eigen::Vector2d>& observations,
    const std::vector<Eigen::Matrix3d>& rotations,
    const std::vector<Eigen::Vector3d>& positions,
    Eigen::Vector3d& point_3d) {

    double point[3] = {point_3d.x(), point_3d.y(), point_3d.z()};

    ceres::Problem problem;
    for (size_t i = 0; i < observations.size(); ++i) {
        auto* cost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 3>(
            new ReprojectionCost(observations[i], rotations[i], positions[i]));
        problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), point);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 20;
    options.linear_solver_type = ceres::DENSE_QR;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.termination_type == ceres::CONVERGENCE ||
        summary.termination_type == ceres::NO_CONVERGENCE) {
        point_3d = Eigen::Vector3d(point[0], point[1], point[2]);
        return point_3d.allFinite();
    }
    return false;
}

} // namespace vio
