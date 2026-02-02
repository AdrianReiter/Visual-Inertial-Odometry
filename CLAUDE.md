# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Configure (from project root)
cmake -S . -B build

# Build all targets
cmake --build build -j$(sysctl -n hw.ncpu)

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single test
./build/test/test_ekf_estimator
./build/test/test_imu_integrator
./build/test/test_feature_tracker
./build/test/test_triangulation
./build/test/test_data_loader

# Run the VIO pipeline (requires TUM-VI dataset)
./build/desktop/vio_main /path/to/dataset-room1_512_16/mav0
./build/desktop/vio_main /path/to/dataset-room1_512_16/mav0 --no-viz
```

## Dependencies

All found via CMake `find_package`. Eigen3, OpenCV 4, and Ceres 2 are from Homebrew (`/opt/homebrew`). Pangolin is built from source and installed to `~/.local` (added to `CMAKE_PREFIX_PATH` in root CMakeLists.txt).

Pangolin was built with `-DBUILD_PANGOLIN_FFMPEG=OFF` to avoid ffmpeg API incompatibilities.

## Architecture

The project is split into three CMake subdirectories:

- **`core/`** — Static library `vio_core` containing all VIO algorithms. Platform-agnostic (no visualization code). This is the library that both the desktop app and a future Android app link against.
- **`desktop/`** — Executable `vio_main` that adds Pangolin 3D visualization and TUM-VI dataset loading. The Pangolin render loop runs on the main thread; the VIO pipeline runs in a background `std::thread`.
- **`test/`** — Five assert-based test executables registered with CTest. Synthetic test data lives in `test/test_data/` (TUM-VI format CSVs + sensor.yaml).

### Core Pipeline Data Flow

`Pipeline::run()` is the main loop. For each camera frame it:
1. Collects IMU measurements between the previous and current frame timestamps
2. Calls `EkfEstimator::predict()` — IMU propagation updates position/velocity/orientation and grows the 15×15 error-state covariance
3. Calls `FeatureTracker::process_frame()` — Shi-Tomasi detection + KLT optical flow tracking + RANSAC outlier rejection via fundamental matrix
4. Stores the current camera pose in the EKF's history ring buffer (max 50 poses)
5. Calls `EkfEstimator::update()` with "lost" features (tracked ≥3 frames then lost) — triangulates each feature via DLT + Ceres refinement, computes reprojection Jacobian, applies Kalman update with Joseph form
6. Invokes `PoseCallback` so the visualizer can update

### EKF Design

Error-state (indirect) EKF with 15D error state: position(3), velocity(3), orientation(3, angle-axis), gyro bias(3), accel bias(3). Nominal state uses quaternion for orientation. The `StateIndex` enum defines offsets into the error vector.

Key files: `ekf_estimator.h/.cpp` (predict/update/inject), `imu_integrator.h/.cpp` (F/G/Q matrices, Euler integration), `triangulation.h/.cpp` (DLT + Ceres `AutoDiffCostFunction`).

### Threading Model

The desktop app runs the Pangolin `render_loop()` on the main thread (required by macOS for OpenGL context). `Pipeline::run()` executes in a `std::thread`. The `Visualizer` uses a `std::mutex` to protect shared trajectory/feature data between the pipeline thread and the render thread.

## Conventions

- All code is in `namespace vio`.
- Timestamps are `uint64_t` nanoseconds (matching TUM-VI convention).
- Quaternion convention: Eigen `Quaterniond(w, x, y, z)` constructor. TUM-VI ground truth uses `[qw, qx, qy, qz]` column order. Beware: `Eigen::Quaterniond::coeffs()` returns `[x, y, z, w]`.
- Camera extrinsic `T_cam_imu` is parsed from `cam0/sensor.yaml` as a row-major 4×4 matrix in the `data:` field.
- Feature observations are stored as undistorted normalized camera coordinates (not pixel coordinates). The `FeatureTracker::undistort_point()` method handles the conversion using `cv::undistortPoints`.
- Covariance symmetry is enforced after every update: `P = 0.5 * (P + P^T)`.

## Dataset Format (TUM-VI)

The pipeline expects the EuRoC-format directory layout under a `mav0/` root:
- `imu0/data.csv` — columns: `timestamp_ns, wx, wy, wz, ax, ay, az`
- `cam0/data.csv` — columns: `timestamp_ns, filename`
- `cam0/data/` — PNG image files
- `cam0/sensor.yaml` — camera intrinsics, distortion, and `T_BS` extrinsic
- `state_groundtruth_estimate0/data.csv` — 17 columns: timestamp, position(3), quaternion(4), velocity(3), gyro_bias(3), accel_bias(3)

## Implementation Status

Stages 1–5 (desktop pipeline) are complete. Stages 6–9 (Android app for Pixel 9a) are planned but not started. See `IMPLEMENTATION_PLAN.md` for details.
