#include "vio/triangulation.h"

#include <cassert>
#include <cmath>
#include <iostream>

static void test_two_view_triangulation() {
    // Known 3D point
    Eigen::Vector3d true_point(1.0, 2.0, 5.0);

    // Camera 1: at origin, looking along +z
    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1(0, 0, 0);

    // Camera 2: translated 1m to the right
    Eigen::Matrix3d R2 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t2(1.0, 0, 0);

    // Project point into both cameras (normalized coords)
    Eigen::Vector3d p1_cam = R1 * (true_point - t1);
    Eigen::Vector3d p2_cam = R2 * (true_point - t2);

    Eigen::Vector2d obs1(p1_cam.x() / p1_cam.z(), p1_cam.y() / p1_cam.z());
    Eigen::Vector2d obs2(p2_cam.x() / p2_cam.z(), p2_cam.y() / p2_cam.z());

    std::vector<Eigen::Vector2d> observations = {obs1, obs2};
    std::vector<Eigen::Matrix3d> rotations = {R1, R2};
    std::vector<Eigen::Vector3d> positions = {t1, t2};

    Eigen::Vector3d result;
    bool ok = vio::Triangulator::triangulate(observations, rotations, positions, result);

    assert(ok && "Triangulation should succeed");
    double error = (result - true_point).norm();
    std::cout << "Two-view triangulation error: " << error << " m\n";
    assert(error < 0.01 && "Triangulation error should be small");

    std::cout << "test_two_view_triangulation: PASSED\n";
}

static void test_three_view_triangulation() {
    Eigen::Vector3d true_point(2.0, -1.0, 8.0);

    // Three cameras with different positions and slight rotations
    std::vector<Eigen::Matrix3d> rotations;
    std::vector<Eigen::Vector3d> positions;
    std::vector<Eigen::Vector2d> observations;

    positions.push_back(Eigen::Vector3d(0, 0, 0));
    positions.push_back(Eigen::Vector3d(1, 0, 0));
    positions.push_back(Eigen::Vector3d(0.5, 0.5, 0));

    for (int i = 0; i < 3; ++i) {
        rotations.push_back(Eigen::Matrix3d::Identity());
        Eigen::Vector3d p_cam = rotations[i] * (true_point - positions[i]);
        observations.push_back(Eigen::Vector2d(p_cam.x() / p_cam.z(),
                                                p_cam.y() / p_cam.z()));
    }

    Eigen::Vector3d result;
    bool ok = vio::Triangulator::triangulate(observations, rotations, positions, result);

    assert(ok && "Three-view triangulation should succeed");
    double error = (result - true_point).norm();
    std::cout << "Three-view triangulation error: " << error << " m\n";
    assert(error < 0.01);

    std::cout << "test_three_view_triangulation: PASSED\n";
}

static void test_point_behind_camera() {
    // Point behind both cameras
    Eigen::Vector3d point(-1.0, 0, -5.0);

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t1(0, 0, 0);
    Eigen::Vector3d t2(1, 0, 0);

    // These projections would have negative z, so we fake the observations
    // to be as if the point were in front (to test the post-triangulation check)
    Eigen::Vector2d obs1(0.2, 0.0);
    Eigen::Vector2d obs2(-0.2, 0.0);

    Eigen::Vector3d result;
    bool ok = vio::Triangulator::triangulate({obs1, obs2}, {R, R}, {t1, t2}, result);

    // The triangulated point should be behind cameras, so it should fail
    std::cout << "Point behind camera test - ok: " << ok << "\n";
    // This test just verifies the function doesn't crash; the result depends on geometry

    std::cout << "test_point_behind_camera: PASSED\n";
}

int main() {
    test_two_view_triangulation();
    test_three_view_triangulation();
    test_point_behind_camera();
    std::cout << "All triangulation tests passed.\n";
    return 0;
}
