#include "vio/feature_tracker.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace vio {

FeatureTracker::FeatureTracker(const Params& params, const CameraIntrinsics& camera)
    : params_(params), camera_(camera) {}

std::vector<TrackedFeature> FeatureTracker::process_frame(const cv::Mat& image,
                                                           Timestamp timestamp) {
    lost_features_.clear();

    if (prev_image_.empty()) {
        prev_image_ = image.clone();
        detect_new_features(image);
        // Initialize tracks for newly detected features
        for (size_t i = 0; i < prev_points_.size(); ++i) {
            int id = feature_ids_[i];
            TrackedFeature& track = active_tracks_[id];
            track.id = id;
            track.observations.push_back(undistort_point(prev_points_[i]));
            track.timestamps.push_back(timestamp);
        }
        return get_active_tracks();
    }

    // Track existing features via KLT
    track_features(image);

    // Update tracks with new observations
    for (size_t i = 0; i < prev_points_.size(); ++i) {
        int id = feature_ids_[i];
        auto& track = active_tracks_[id];
        track.observations.push_back(undistort_point(prev_points_[i]));
        track.timestamps.push_back(timestamp);
    }

    // Remove features tracked too long
    std::vector<cv::Point2f> kept_points;
    std::vector<int> kept_ids;
    for (size_t i = 0; i < prev_points_.size(); ++i) {
        int id = feature_ids_[i];
        if (active_tracks_[id].observations.size() <=
            static_cast<size_t>(params_.max_track_length)) {
            kept_points.push_back(prev_points_[i]);
            kept_ids.push_back(id);
        } else {
            // Feature tracked too long, retire it
            if (active_tracks_[id].observations.size() >= 3) {
                lost_features_.push_back(active_tracks_[id]);
            }
            active_tracks_.erase(id);
        }
    }
    prev_points_ = kept_points;
    feature_ids_ = kept_ids;

    // Detect new features if we have too few
    if (static_cast<int>(prev_points_.size()) < params_.max_features * 0.7) {
        detect_new_features(image);
        // Initialize new tracks
        for (size_t i = kept_points.size(); i < prev_points_.size(); ++i) {
            int id = feature_ids_[i];
            TrackedFeature& track = active_tracks_[id];
            track.id = id;
            track.observations.push_back(undistort_point(prev_points_[i]));
            track.timestamps.push_back(timestamp);
        }
    }

    prev_image_ = image.clone();
    return get_active_tracks();
}

void FeatureTracker::detect_new_features(const cv::Mat& image) {
    // Create mask to avoid detecting near existing features
    cv::Mat mask = cv::Mat::ones(image.size(), CV_8UC1) * 255;
    for (const auto& pt : prev_points_) {
        cv::circle(mask, pt, static_cast<int>(params_.min_distance), cv::Scalar(0), -1);
    }

    int needed = params_.max_features - static_cast<int>(prev_points_.size());
    if (needed <= 0) return;

    std::vector<cv::Point2f> new_pts;
    cv::goodFeaturesToTrack(image, new_pts, needed, params_.quality_level,
                            params_.min_distance, mask);

    for (const auto& pt : new_pts) {
        prev_points_.push_back(pt);
        feature_ids_.push_back(next_feature_id_++);
    }
}

void FeatureTracker::track_features(const cv::Mat& curr_image) {
    if (prev_points_.empty()) return;

    std::vector<cv::Point2f> curr_points;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(prev_image_, curr_image, prev_points_, curr_points,
                              status, err, params_.win_size,
                              params_.max_pyramid_level);

    // Collect survived features
    std::vector<cv::Point2f> good_prev, good_curr;
    std::vector<int> good_ids;

    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) {
            // Feature lost — add to lost_features if it has enough observations
            int id = feature_ids_[i];
            if (active_tracks_.count(id) &&
                active_tracks_[id].observations.size() >= 3) {
                lost_features_.push_back(active_tracks_[id]);
            }
            active_tracks_.erase(id);
            continue;
        }
        // Check bounds
        if (curr_points[i].x < 0 || curr_points[i].y < 0 ||
            curr_points[i].x >= curr_image.cols || curr_points[i].y >= curr_image.rows) {
            int id = feature_ids_[i];
            if (active_tracks_.count(id) &&
                active_tracks_[id].observations.size() >= 3) {
                lost_features_.push_back(active_tracks_[id]);
            }
            active_tracks_.erase(id);
            continue;
        }
        good_prev.push_back(prev_points_[i]);
        good_curr.push_back(curr_points[i]);
        good_ids.push_back(feature_ids_[i]);
    }

    // RANSAC outlier rejection using fundamental matrix
    if (good_prev.size() >= 8) {
        std::vector<uchar> inlier_mask;
        cv::findFundamentalMat(good_prev, good_curr, cv::FM_RANSAC, 1.0, 0.99,
                               inlier_mask);

        std::vector<cv::Point2f> inlier_points;
        std::vector<int> inlier_ids;
        for (size_t i = 0; i < inlier_mask.size(); ++i) {
            if (inlier_mask[i]) {
                inlier_points.push_back(good_curr[i]);
                inlier_ids.push_back(good_ids[i]);
            } else {
                int id = good_ids[i];
                if (active_tracks_.count(id) &&
                    active_tracks_[id].observations.size() >= 3) {
                    lost_features_.push_back(active_tracks_[id]);
                }
                active_tracks_.erase(id);
            }
        }
        prev_points_ = inlier_points;
        feature_ids_ = inlier_ids;
    } else {
        prev_points_ = good_curr;
        feature_ids_ = good_ids;
    }
}

Eigen::Vector2d FeatureTracker::undistort_point(const cv::Point2f& pt) {
    if (camera_.fx == 0) {
        return Eigen::Vector2d(pt.x, pt.y);
    }

    cv::Mat pts_in(1, 1, CV_64FC2);
    pts_in.at<cv::Vec2d>(0) = cv::Vec2d(pt.x, pt.y);

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        camera_.fx, 0, camera_.cx,
        0, camera_.fy, camera_.cy,
        0, 0, 1);

    cv::Mat D;
    // TUM-VI usually provides 4 coefficients for the equidistant model
    if (camera_.distortion.size() == 4) {
        D = (cv::Mat_<double>(4, 1) <<
            camera_.distortion[0], camera_.distortion[1],
            camera_.distortion[2], camera_.distortion[3]);
    } else {
        // Fallback or handle other sizes
        // Note: Check if your vector parsing needs to handle rows/cols differently
    }

    cv::Mat pts_out;
    
    // CHANGED: Use fisheye undistort with P=identity (empty) to get normalized coordinates
    cv::fisheye::undistortPoints(pts_in, pts_out, K, D); 
    // WAS: cv::undistortPoints(pts_in, pts_out, K, D);

    cv::Vec2d p = pts_out.at<cv::Vec2d>(0);
    return Eigen::Vector2d(p[0], p[1]);
}

std::vector<TrackedFeature> FeatureTracker::get_active_tracks() {
    std::vector<TrackedFeature> result;
    result.reserve(active_tracks_.size());
    for (const auto& [id, track] : active_tracks_) {
        result.push_back(track);
    }
    return result;
}

} // namespace vio
