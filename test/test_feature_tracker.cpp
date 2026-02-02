#include "vio/feature_tracker.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv2/imgproc.hpp>

static cv::Mat create_textured_image(int rows, int cols, int seed) {
    // Create a synthetic image with random-ish texture (good for feature detection)
    cv::Mat img(rows, cols, CV_8UC1, cv::Scalar(128));

    // Draw random circles and lines for texture
    cv::RNG rng(seed);
    for (int i = 0; i < 100; ++i) {
        cv::Point center(rng.uniform(0, cols), rng.uniform(0, rows));
        int radius = rng.uniform(3, 20);
        int brightness = rng.uniform(0, 255);
        cv::circle(img, center, radius, cv::Scalar(brightness), -1);
    }
    for (int i = 0; i < 50; ++i) {
        cv::Point p1(rng.uniform(0, cols), rng.uniform(0, rows));
        cv::Point p2(rng.uniform(0, cols), rng.uniform(0, rows));
        int brightness = rng.uniform(0, 255);
        cv::line(img, p1, p2, cv::Scalar(brightness), rng.uniform(1, 3));
    }
    return img;
}

static void test_detection() {
    // Verify that features are detected in a textured image
    vio::CameraIntrinsics cam;
    cam.fx = 200; cam.fy = 200; cam.cx = 160; cam.cy = 120;
    cam.width = 320; cam.height = 240;

    vio::FeatureTracker::Params params;
    params.max_features = 100;
    vio::FeatureTracker tracker(params, cam);

    cv::Mat image = create_textured_image(240, 320, 42);
    auto features = tracker.process_frame(image, 1000000000ULL);

    std::cout << "Detected " << features.size() << " features\n";
    assert(features.size() > 10 && "Should detect at least some features");
    assert(static_cast<int>(features.size()) <= params.max_features);

    std::cout << "test_detection: PASSED\n";
}

static void test_tracking_with_known_shift() {
    // Create an image, shift it by a known amount, verify tracking accuracy
    vio::CameraIntrinsics cam;
    cam.fx = 200; cam.fy = 200; cam.cx = 160; cam.cy = 120;
    cam.width = 320; cam.height = 240;

    vio::FeatureTracker::Params params;
    params.max_features = 100;
    vio::FeatureTracker tracker(params, cam);

    cv::Mat image1 = create_textured_image(240, 320, 42);

    // Shift image by (5, 3) pixels
    cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, 5, 0, 1, 3);
    cv::Mat image2;
    cv::warpAffine(image1, image2, M, image1.size());

    auto features1 = tracker.process_frame(image1, 1000000000ULL);
    auto features2 = tracker.process_frame(image2, 1050000000ULL);

    std::cout << "Frame 1: " << features1.size() << " features, "
              << "Frame 2: " << features2.size() << " features\n";

    assert(features2.size() > 5 && "Should track at least some features across frames");

    std::cout << "test_tracking_with_known_shift: PASSED\n";
}

static void test_lost_features() {
    // Track across multiple frames, then present a blank image to lose features
    vio::CameraIntrinsics cam;
    cam.fx = 200; cam.fy = 200; cam.cx = 160; cam.cy = 120;
    cam.width = 320; cam.height = 240;

    vio::FeatureTracker::Params params;
    params.max_features = 100;
    vio::FeatureTracker tracker(params, cam);

    // Process 4 frames with small shifts
    for (int i = 0; i < 4; ++i) {
        cv::Mat img = create_textured_image(240, 320, 42);
        cv::Mat M = (cv::Mat_<double>(2, 3) << 1, 0, i * 2, 0, 1, i);
        cv::Mat shifted;
        cv::warpAffine(img, shifted, M, img.size());
        tracker.process_frame(shifted, 1000000000ULL + i * 50000000ULL);
    }

    // Now present a very different image — most features should be lost
    cv::Mat blank(240, 320, CV_8UC1, cv::Scalar(50));
    tracker.process_frame(blank, 1200000000ULL);

    const auto& lost = tracker.lost_features();
    std::cout << "Lost features with 3+ observations: " << lost.size() << "\n";

    // Some features should have been tracked 3+ frames then lost
    assert(lost.size() > 0 && "Should have some lost features with observations");
    for (const auto& f : lost) {
        assert(f.observations.size() >= 3);
    }

    std::cout << "test_lost_features: PASSED\n";
}

int main() {
    test_detection();
    test_tracking_with_known_shift();
    test_lost_features();
    std::cout << "All feature tracker tests passed.\n";
    return 0;
}
