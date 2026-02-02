## Stage 1: Project Skeleton + Data Loading
**Goal**: CMakeLists, types, TUM-VI data loader, desktop main
**Success Criteria**: Compiles, reads CSV data, prints statistics
**Tests**: test_data_loader
**Status**: Complete

## Stage 2: IMU Integration
**Goal**: IMU state propagation with covariance
**Success Criteria**: Dead reckoning matches expected behavior
**Tests**: test_imu_integrator (stationary, const accel, const rotation)
**Status**: Complete

## Stage 3: Feature Tracking
**Goal**: Shi-Tomasi + KLT with RANSAC outlier rejection
**Success Criteria**: Features tracked across frames, lost features collected
**Tests**: test_feature_tracker
**Status**: Complete

## Stage 4: EKF + Triangulation
**Goal**: Error-state EKF with IMU prediction and visual measurement update
**Success Criteria**: Covariance decreases after update, triangulation accurate
**Tests**: test_triangulation, test_ekf_estimator
**Status**: Complete

## Stage 5: Desktop Visualization (Pangolin)
**Goal**: 3D trajectory visualization (estimated vs ground truth)
**Success Criteria**: Pangolin window renders trajectories and camera frustum
**Tests**: Visual verification
**Status**: Complete

## Stage 6: Android Project Scaffolding
**Goal**: Kotlin app with JNI bridge to C++ core
**Status**: Not Started

## Stage 7: Camera + IMU Pipeline (Android)
**Goal**: Camera2 API + SensorManager feeding data to C++ via JNI
**Status**: Not Started

## Stage 8: Wire VIO Core (Android)
**Goal**: Full VIO pipeline running on Pixel 9a
**Status**: Not Started

## Stage 9: Android Visualization
**Goal**: OpenGL ES 2.0 trajectory rendering on device
**Status**: Not Started
