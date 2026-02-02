# Visual Inertial Odometry (VIO) Project

This project implements a monocular Visual Inertial Odometry system from scratch using C++17. It fuses IMU measurements with visual features using an Error-State Extended Kalman Filter (ES-EKF) to estimate the trajectory of a sensor.

## Project Overview

The system is designed with a modular architecture, separating the core VIO algorithms from the visualization and platform-specific code.

*   **Language:** C++17
*   **Build System:** CMake
*   **Core Libraries:**
    *   **Eigen3:** Linear algebra and geometry (quaternions).
    *   **OpenCV 4:** Image processing, feature detection/tracking (KLT), and camera calibration/undistortion.
    *   **Ceres Solver:** Non-linear least squares optimization (for triangulation).
*   **Visualization:** Pangolin (for the desktop application).

### Architecture

The codebase is organized into three main modules:

1.  **`core/`**: A platform-agnostic static library (`vio_core`) containing the VIO logic.
    *   `Pipeline`: Orchestrates the system (data loading, IMU integration, tracking, EKF).
    *   `EkfEstimator`: Implements the 15-state ES-EKF (Position, Velocity, Orientation, Gyro Bias, Accel Bias).
    *   `FeatureTracker`: Handles Shi-Tomasi detection, KLT tracking, and RANSAC outlier rejection.
    *   `ImuIntegrator`: Propagates the state using IMU data.
    *   `DataLoader`: Parses TUM-VI (EuRoC) format datasets.
2.  **`desktop/`**: The desktop executable (`vio_main`) that links against `vio_core`.
    *   `Visualizer`: Renders the 3D trajectory and features using Pangolin.
    *   `main.cpp`: Entry point; runs the pipeline in a background thread and the rendering loop on the main thread.
3.  **`test/`**: Unit tests for individual components using CTest.

## Building and Running

### Prerequisites
*   CMake >= 3.16
*   C++17 compiler
*   **Libraries:** Eigen3, OpenCV 4, Ceres Solver (available via Homebrew on macOS).
*   **Pangolin:** Must be installed (usually built from source) and found by CMake.

### Build Commands

```bash
# Configure the project
cmake -S . -B build

# Build all targets (optimized parallel build)
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Running the Desktop App

The application requires a dataset in the TUM-VI / EuRoC format (e.g., `dataset-room1_512_16`).

```bash
# Run with visualization (Pangolin window)
./build/desktop/vio_main /path/to/dataset-room1_512_16/mav0

# Run without visualization (console output only)
./build/desktop/vio_main /path/to/dataset-room1_512_16/mav0 --no-viz
```

### Running Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run a specific test
./build/test/test_ekf_estimator
```

## Development Conventions

*   **Namespace:** All core code is within the `vio` namespace.
*   **Coordinates:**
    *   Timestamps are in nanoseconds (`uint64_t`).
    *   Ground truth data is expected in the TUM-VI format.
    *   Camera extrinsics (`T_BS`) are read from `sensor.yaml`.
*   **Math:**
    *   Quaternions use `Eigen::Quaterniond(w, x, y, z)`. **Note:** `coeffs()` stores as `[x, y, z, w]`.
    *   Covariance matrices are explicitly symmetrized after updates.
*   **Code Style:**
    *   Modern C++17 features are encouraged.
    *   Standard CMake patterns are used for dependency management.
