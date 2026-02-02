#pragma once

#include "vio/types.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <mutex>
#include <thread>
#include <vector>

namespace vio {

class Visualizer {
public:
    Visualizer();
    ~Visualizer();

    void start();

    void render_loop(); 

    void update_estimated_pose(const Eigen::Vector3d& position,
                                const Eigen::Quaterniond& orientation);
    void update_ground_truth(const Eigen::Vector3d& position);
    void update_features(const std::vector<Eigen::Vector3d>& points);

    bool should_quit() const;

private:
    void draw_trajectory(const std::vector<Eigen::Vector3d>& traj,
                         float r, float g, float b);
    void draw_camera_frustum(const Eigen::Vector3d& pos,
                              const Eigen::Quaterniond& ori,
                              float r, float g, float b, float size);
    void draw_grid(float size, float step);

    std::thread render_thread_;
    mutable std::mutex data_mutex_;

    std::vector<Eigen::Vector3d> estimated_trajectory_;
    std::vector<Eigen::Vector3d> ground_truth_trajectory_;
    std::vector<Eigen::Vector3d> current_features_;
    Eigen::Vector3d current_position_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond current_orientation_ = Eigen::Quaterniond::Identity();
    bool quit_ = false;
    bool started_ = false;
};

} // namespace vio
