# Visual Inertial Odometry

A C++17 implementation of a monocular Visual Inertial Odometry (VIO) system using an Error-State Extended Kalman Filter (ES-EKF). This project fuses inertial measurements with visual features to estimate a 6-DOF trajectory in real-time.

## Overview

The system processes monocular camera frames and IMU data to estimate position, velocity, and orientation. It utilizes a 15-state indirect EKF to track errors in the state and biases in the inertial sensors.

### Core Components

*   **State Estimation**: 15-state Error-State EKF (Position, Velocity, Orientation, Gyro Bias, Accel Bias).
*   **Feature Tracking**: Shi-Tomasi corner detection with KLT optical flow and RANSAC-based outlier rejection.
*   **IMU Integration**: High-frequency state propagation with covariance estimation.
*   **Triangulation**: DLT with non-linear refinement via Ceres Solver for lost features.
*   **Visualization**: Real-time 3D trajectory and feature point cloud rendering using Pangolin.

## Dependencies

*   CMake (>= 3.16)
*   Eigen3
*   OpenCV 4
*   Ceres Solver
*   Pangolin

## Building

```bash
cmake -S . -B build
cmake --build build -j$(sysctl -n hw.ncpu)
```

## Usage

The application expects datasets in the TUM-VI or EuRoC format.

```bash
# Run with 3D visualization
./build/desktop/vio_main /path/to/dataset/mav0

# Run in headless mode
./build/desktop/vio_main /path/to/dataset/mav0 --no-viz
```

## Testing

Comprehensive unit tests are provided for core components:

```bash
ctest --test-dir build --output-on-failure
```

## Architecture

*   `core/`: Platform-agnostic library containing all VIO algorithms.
*   `desktop/`: Desktop entry point and OpenGL-based visualization.
*   `test/`: Unit tests and synthetic test data.
