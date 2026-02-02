#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace vio {

class Triangulator {
public:
    // Triangulate a 3D point from multiple observations.
    // observations: normalized camera coordinates (undistorted)
    // rotations: R_cam_world for each observation
    // positions: camera position in world frame for each observation
    // Returns false if triangulation fails (degenerate geometry, point behind cameras, etc.)
    static bool triangulate(
        const std::vector<Eigen::Vector2d>& observations,
        const std::vector<Eigen::Matrix3d>& rotations,
        const std::vector<Eigen::Vector3d>& positions,
        Eigen::Vector3d& point_3d);

private:
    static Eigen::Vector3d triangulate_linear(
        const std::vector<Eigen::Vector2d>& observations,
        const std::vector<Eigen::Matrix3d>& rotations,
        const std::vector<Eigen::Vector3d>& positions);

    static bool refine_ceres(
        const std::vector<Eigen::Vector2d>& observations,
        const std::vector<Eigen::Matrix3d>& rotations,
        const std::vector<Eigen::Vector3d>& positions,
        Eigen::Vector3d& point_3d);
};

} // namespace vio
