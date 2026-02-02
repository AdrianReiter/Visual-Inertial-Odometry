#include "vio/pipeline.h"

#include <cmath>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace vio {

Pipeline::Pipeline(const Config& config)
    : config_(config), data_loader_(config.dataset_path) {}

void Pipeline::run(PoseCallback pose_callback) {
    std::cout << "Loading dataset...\n";
    if (!data_loader_.load()) {
        std::cerr << "Failed to load dataset\n";
        return;
    }

    const auto& images = data_loader_.cam0_images();
    const auto& imu = data_loader_.imu_data();
    const auto& cam = data_loader_.camera_intrinsics();

    std::cout << "Images: " << images.size() << ", IMU: " << imu.size() << "\n";

    // Set up feature tracker
    FeatureTracker::Params ft_params;
    ft_params.max_features = 300;      // Increase count
    ft_params.quality_level = 0.001;   // LOWER this significantly (default is often 0.01)
    ft_params.min_distance = 15;       // Reduce spacing to find more points
    
    // Optional: Reduce win_size if fast motion is blurring features
    // ft_params.win_size = cv::Size(15, 15);
    feature_tracker_ = std::make_unique<FeatureTracker>(ft_params, cam);


    // Set up EKF
    EkfEstimator::Params ekf_params;
    ekf_params.feature_noise_std = 1.5;
    ekf_ = std::make_unique<EkfEstimator>(ekf_params, cam);

    // Initialize from ground truth
    initialize_from_groundtruth();

    Timestamp prev_frame_time = images.front().timestamp;
    double total_error = 0.0;
    int error_count = 0;

    for (size_t frame_idx = 0; frame_idx < images.size(); ++frame_idx) {
        Timestamp t = images[frame_idx].timestamp;

        // 1. IMU prediction
        if (frame_idx > 0) {
            auto imu_meas = get_imu_between(prev_frame_time, t);
            ekf_->predict(imu_meas, t);
        }

        // 2. Load image
        cv::Mat image = cv::imread(images[frame_idx].filepath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            if (frame_idx < 5) {
                std::cerr << "Warning: could not load image "
                          << images[frame_idx].filepath << "\n";
            }
            prev_frame_time = t;
            continue;
        }

        // 3. Track features
        feature_tracker_->process_frame(image, t);

        // 4. Store camera pose for triangulation
        ekf_->store_camera_pose(t);

        // 5. EKF update with lost features
        const auto& lost = feature_tracker_->lost_features();
        if (!lost.empty()) {
            ekf_->update(lost);
        }

        // 6. Compute error vs ground truth
        const GroundTruthPose* gt = find_nearest_groundtruth(t);
        if (gt) {
            double pos_error = (ekf_->state().position - gt->position).norm();
            total_error += pos_error * pos_error;
            error_count++;
        }

        // 7. Callback
        if (pose_callback) {
            // Collect triangulated feature 3D points (from lost features)
            std::vector<Eigen::Vector3d> feature_points;
            for (const auto& f : lost) {
                if (f.is_triangulated) {
                    feature_points.push_back(f.position_3d);
                }
            }
            pose_callback(ekf_->state(), gt, feature_points, image);
        }

        // Print progress every 100 frames
        if (frame_idx % 100 == 0 && error_count > 0) {
            double rmse = std::sqrt(total_error / error_count);
            std::cout << "Frame " << frame_idx << "/" << images.size()
                      << " | RMSE: " << rmse << " m"
                      << " | Features tracked: " << lost.size() << " lost\n";
        }

        prev_frame_time = t;
    }

    if (error_count > 0) {
        double final_rmse = std::sqrt(total_error / error_count);
        std::cout << "\n=== Final Results ===\n"
                  << "Processed " << images.size() << " frames\n"
                  << "Position RMSE: " << final_rmse << " m\n";
    }
}

void Pipeline::initialize_from_groundtruth() {
    const auto& gt = data_loader_.ground_truth();
    const auto& images = data_loader_.cam0_images();

    if (gt.empty() || images.empty()) {
        std::cerr << "No ground truth or images to initialize from\n";
        return;
    }

    // Find the ground truth closest to the first image timestamp
    const GroundTruthPose* init_gt = find_nearest_groundtruth(images.front().timestamp);
    if (!init_gt) {
        std::cerr << "Could not find matching ground truth for initialization\n";
        return;
    }

    State initial_state;
    initial_state.position = init_gt->position;
    initial_state.velocity = init_gt->velocity;
    initial_state.orientation = init_gt->orientation;
    initial_state.gyro_bias = init_gt->gyro_bias;
    initial_state.accel_bias = init_gt->accel_bias;
    initial_state.timestamp = images.front().timestamp;

    // Initial covariance: small uncertainty since we initialize from ground truth
    initial_state.covariance = Eigen::Matrix<double, 15, 15>::Zero();
    initial_state.covariance.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * 0.01;  // pos
    initial_state.covariance.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 0.01;  // vel
    initial_state.covariance.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity() * 0.01;  // ori
    initial_state.covariance.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity() * 1e-4;  // bg
    initial_state.covariance.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * 1e-4; // ba

    ekf_->initialize(initial_state);

    std::cout << "Initialized from ground truth at t="
              << init_gt->timestamp * 1e-9 << "s\n"
              << "  pos: [" << init_gt->position.transpose() << "]\n"
              << "  vel: [" << init_gt->velocity.transpose() << "]\n";
}

const GroundTruthPose* Pipeline::find_nearest_groundtruth(Timestamp t) const {
    const auto& gt = data_loader_.ground_truth();
    if (gt.empty()) return nullptr;

    const GroundTruthPose* best = nullptr;
    uint64_t best_diff = UINT64_MAX;

    for (const auto& pose : gt) {
        uint64_t diff = (pose.timestamp > t)
                            ? (pose.timestamp - t)
                            : (t - pose.timestamp);
        if (diff < best_diff) {
            best_diff = diff;
            best = &pose;
        }
        // Early exit if we've passed the target time
        if (pose.timestamp > t && diff > best_diff) break;
    }

    // Allow up to 50ms tolerance
    if (best && best_diff < 50000000ULL) return best;
    return nullptr;
}

std::vector<ImuMeasurement> Pipeline::get_imu_between(Timestamp t0,
                                                       Timestamp t1) const {
    const auto& imu = data_loader_.imu_data();
    std::vector<ImuMeasurement> result;

    for (const auto& m : imu) {
        if (m.timestamp < t0) continue;
        if (m.timestamp > t1) break;
        result.push_back(m);
    }
    return result;
}

} // namespace vio
