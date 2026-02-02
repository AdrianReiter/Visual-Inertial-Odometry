#include "visualizer.h"

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/handler/handler.h>

namespace vio {

Visualizer::Visualizer() = default;

Visualizer::~Visualizer() {
    quit_ = true;
    if (render_thread_.joinable()) {
        render_thread_.join();
    }
}

void Visualizer::start() {
    if (started_) return;
    started_ = true;
    render_thread_ = std::thread(&Visualizer::render_loop, this);
}

void Visualizer::update_estimated_pose(const Eigen::Vector3d& position,
                                        const Eigen::Quaterniond& orientation) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_position_ = position;
    current_orientation_ = orientation;
    estimated_trajectory_.push_back(position);
}

void Visualizer::update_ground_truth(const Eigen::Vector3d& position) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    ground_truth_trajectory_.push_back(position);
}

void Visualizer::update_features(const std::vector<Eigen::Vector3d>& points) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    current_features_ = points;
}

bool Visualizer::should_quit() const {
    return quit_;
}

void Visualizer::render_loop() {
    pangolin::CreateWindowAndBind("VIO Trajectory", 1280, 720);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1280, 720, 500, 500, 640, 360, 0.1, 100),
        pangolin::ModelViewLookAt(0, -3, -5, 0, 0, 0, pangolin::AxisNegY));

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit() && !quit_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

        d_cam.Activate(s_cam);

        // Copy data under lock
        std::vector<Eigen::Vector3d> est_traj, gt_traj, features;
        Eigen::Vector3d pos;
        Eigen::Quaterniond ori;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            est_traj = estimated_trajectory_;
            gt_traj = ground_truth_trajectory_;
            features = current_features_;
            pos = current_position_;
            ori = current_orientation_;
        }

        // Draw ground grid
        draw_grid(20.0f, 1.0f);

        // Draw ground truth trajectory (red)
        draw_trajectory(gt_traj, 0.8f, 0.2f, 0.2f);

        // Draw estimated trajectory (green)
        draw_trajectory(est_traj, 0.2f, 0.8f, 0.2f);

        // Draw current camera frustum (green)
        draw_camera_frustum(pos, ori, 0.2f, 0.8f, 0.2f, 0.2f);

        // Draw 3D feature points (blue)
        if (!features.empty()) {
            glPointSize(3.0f);
            glColor3f(0.3f, 0.3f, 0.9f);
            glBegin(GL_POINTS);
            for (const auto& p : features) {
                glVertex3d(p.x(), p.y(), p.z());
            }
            glEnd();
        }

        pangolin::FinishFrame();
    }

    quit_ = true;
    pangolin::DestroyWindow("VIO Trajectory");
}

void Visualizer::draw_trajectory(const std::vector<Eigen::Vector3d>& traj,
                                  float r, float g, float b) {
    if (traj.size() < 2) return;

    glLineWidth(2.0f);
    glColor3f(r, g, b);
    glBegin(GL_LINE_STRIP);
    for (const auto& p : traj) {
        glVertex3d(p.x(), p.y(), p.z());
    }
    glEnd();
}

void Visualizer::draw_camera_frustum(const Eigen::Vector3d& pos,
                                      const Eigen::Quaterniond& ori,
                                      float r, float g, float b, float size) {
    Eigen::Matrix3d R = ori.toRotationMatrix();

    // Frustum corners in camera frame
    float w = size * 0.5f;
    float h = size * 0.4f;
    float z = size;

    Eigen::Vector3d p0 = pos;
    Eigen::Vector3d p1 = pos + R * Eigen::Vector3d(-w, -h, z);
    Eigen::Vector3d p2 = pos + R * Eigen::Vector3d( w, -h, z);
    Eigen::Vector3d p3 = pos + R * Eigen::Vector3d( w,  h, z);
    Eigen::Vector3d p4 = pos + R * Eigen::Vector3d(-w,  h, z);

    glLineWidth(1.5f);
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    // Edges from center to corners
    auto v = [](const Eigen::Vector3d& p) { glVertex3d(p.x(), p.y(), p.z()); };
    v(p0); v(p1);
    v(p0); v(p2);
    v(p0); v(p3);
    v(p0); v(p4);
    // Rectangle
    v(p1); v(p2);
    v(p2); v(p3);
    v(p3); v(p4);
    v(p4); v(p1);
    glEnd();
}

void Visualizer::draw_grid(float size, float step) {
    glLineWidth(0.5f);
    glColor4f(0.3f, 0.3f, 0.3f, 0.5f);
    glBegin(GL_LINES);
    for (float i = -size; i <= size; i += step) {
        glVertex3f(i, 0, -size);
        glVertex3f(i, 0, size);
        glVertex3f(-size, 0, i);
        glVertex3f(size, 0, i);
    }
    glEnd();
}

} // namespace vio
