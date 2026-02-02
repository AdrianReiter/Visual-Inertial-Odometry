add_library(vio_core
    src/data_loader.cpp
    src/imu_integrator.cpp
    src/feature_tracker.cpp
    src/triangulation.cpp
    src/ekf_estimator.cpp
    src/pipeline.cpp
)

target_include_directories(vio_core PUBLIC include)

target_link_libraries(vio_core PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    Ceres::ceres
)
