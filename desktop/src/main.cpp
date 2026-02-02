#include "vio/pipeline.h"
#include "visualizer.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <thread> 

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: vio_main <dataset_path> [--no-viz]\n"
                  << "  e.g. vio_main /path/to/dataset-room1_512_16/mav0\n";
        return 1;
    }

    bool visualize = true;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-viz") {
            visualize = false;
        }
    }

    vio::Pipeline::Config config;
    config.dataset_path = argv[1];
    config.visualize = visualize;

    vio::Pipeline pipeline(config);

    std::cout << std::fixed << std::setprecision(4);

    if (visualize) {
        vio::Visualizer viz;
        // 1. Launch the Pipeline in a background thread
        std::thread pipeline_thread([&]() {
            pipeline.run([&viz](const vio::State& state,
                                const vio::GroundTruthPose* gt,
                                const std::vector<Eigen::Vector3d>& features,
                                const cv::Mat& /*image*/) {
                // Check if the window was closed to stop processing early (optional)
                if (viz.should_quit()) return; 

                viz.update_estimated_pose(state.position, state.orientation);
                if (gt) {
                    viz.update_ground_truth(gt->position);
                }
                if (!features.empty()) {
                    viz.update_features(features);
                }
            });
        });

        // 2. Run the Visualization loop on the Main Thread (Blocking)
        viz.render_loop();

        // 3. Clean up: Wait for pipeline to finish if window is closed
        if (pipeline_thread.joinable()) {
            pipeline_thread.join();
        }
        
    } else {
        pipeline.run();
    }

    return 0;
}
