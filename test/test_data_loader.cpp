#include "vio/data_loader.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

static void test_imu_parsing() {
    // Create a temporary dataset structure mimicking TUM-VI
    auto tmp = fs::temp_directory_path() / "vio_test_dataset";
    fs::create_directories(tmp / "imu0");
    fs::create_directories(tmp / "cam0" / "data");
    fs::create_directories(tmp / "state_groundtruth_estimate0");

    // Copy test data files
    auto test_dir = fs::path(TEST_DATA_DIR);
    fs::copy_file(test_dir / "imu_data.csv", tmp / "imu0" / "data.csv",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(test_dir / "cam0_data.csv", tmp / "cam0" / "data.csv",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(test_dir / "groundtruth.csv",
                  tmp / "state_groundtruth_estimate0" / "data.csv",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(test_dir / "sensor.yaml", tmp / "cam0" / "sensor.yaml",
                  fs::copy_options::overwrite_existing);

    vio::DataLoader loader(tmp.string());
    assert(loader.load() && "DataLoader::load() failed");

    // Verify IMU data
    const auto& imu = loader.imu_data();
    assert(imu.size() == 3);
    assert(imu[0].timestamp == 1000000000ULL);
    assert(std::abs(imu[0].angular_velocity.x() - 0.01) < 1e-9);
    assert(std::abs(imu[0].angular_velocity.y() - (-0.02)) < 1e-9);
    assert(std::abs(imu[0].linear_acceleration.z() - 9.81) < 1e-9);
    assert(imu[1].timestamp == 1005000000ULL);
    assert(imu[2].timestamp == 1010000000ULL);

    // Verify camera data
    const auto& images = loader.cam0_images();
    assert(images.size() == 3);
    assert(images[0].timestamp == 1000000000ULL);
    assert(images[0].filepath.find("1000000000.png") != std::string::npos);

    // Verify ground truth
    const auto& gt = loader.ground_truth();
    assert(gt.size() == 2);
    assert(gt[0].timestamp == 1000000000ULL);
    assert(std::abs(gt[0].position.x() - 1.0) < 1e-9);
    assert(std::abs(gt[0].position.y() - 2.0) < 1e-9);
    assert(std::abs(gt[0].position.z() - 3.0) < 1e-9);
    assert(std::abs(gt[0].orientation.w() - 1.0) < 1e-6);
    assert(std::abs(gt[0].velocity.x() - 0.5) < 1e-9);
    assert(std::abs(gt[0].gyro_bias.x() - 0.001) < 1e-9);
    assert(std::abs(gt[0].accel_bias.x() - 0.01) < 1e-9);

    // Verify camera intrinsics
    const auto& cam = loader.camera_intrinsics();
    assert(cam.width == 512);
    assert(cam.height == 512);
    assert(std::abs(cam.fx - 190.978) < 1e-3);
    assert(std::abs(cam.fy - 186.145) < 1e-3);
    assert(std::abs(cam.cx - 254.931) < 1e-3);
    assert(std::abs(cam.cy - 256.897) < 1e-3);
    assert(cam.distortion.size() == 4);
    assert(std::abs(cam.distortion[0] - 0.0034) < 1e-6);

    // Verify extrinsic (first element of T_BS)
    assert(std::abs(cam.T_cam_imu(0, 0) - (-0.9995)) < 1e-4);

    // Cleanup
    fs::remove_all(tmp);

    std::cout << "test_imu_parsing: PASSED\n";
}

int main() {
    test_imu_parsing();
    std::cout << "All data loader tests passed.\n";
    return 0;
}
