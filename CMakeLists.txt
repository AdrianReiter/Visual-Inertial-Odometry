cmake_minimum_required(VERSION 3.16)
project(VisualInertialOdometry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Homebrew on Apple Silicon
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc features2d calib3d video highgui)
find_package(Ceres 2 REQUIRED)

add_subdirectory(core)
add_subdirectory(desktop)

enable_testing()
add_subdirectory(test)
