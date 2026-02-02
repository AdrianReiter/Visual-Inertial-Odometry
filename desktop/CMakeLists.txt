find_package(Pangolin REQUIRED)

add_executable(vio_main
    src/main.cpp
    src/visualizer.cpp
)
target_include_directories(vio_main PRIVATE include)
target_link_libraries(vio_main PRIVATE vio_core pango_display pango_opengl)
