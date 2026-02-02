#include "vio/data_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace vio {

DataLoader::DataLoader(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

bool DataLoader::load() {
    bool ok = true;
    ok &= parse_imu_csv(dataset_path_ + "/imu0/data.csv");
    ok &= parse_image_csv(dataset_path_ + "/cam0/data.csv", cam0_images_,
                           dataset_path_ + "/cam0/data/");
    ok &= parse_groundtruth_csv(
        dataset_path_ + "/state_groundtruth_estimate0/data.csv");
    ok &= parse_sensor_yaml(dataset_path_ + "/cam0/sensor.yaml");
    return ok;
}

bool DataLoader::parse_imu_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open IMU file: " << filepath << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;

        ImuMeasurement m;
        std::getline(ss, token, ','); m.timestamp = std::stoull(token);
        std::getline(ss, token, ','); m.angular_velocity.x() = std::stod(token);
        std::getline(ss, token, ','); m.angular_velocity.y() = std::stod(token);
        std::getline(ss, token, ','); m.angular_velocity.z() = std::stod(token);
        std::getline(ss, token, ','); m.linear_acceleration.x() = std::stod(token);
        std::getline(ss, token, ','); m.linear_acceleration.y() = std::stod(token);
        std::getline(ss, token, ','); m.linear_acceleration.z() = std::stod(token);

        imu_data_.push_back(m);
    }
    return true;
}

bool DataLoader::parse_image_csv(const std::string& filepath,
                                  std::vector<ImageData>& out,
                                  const std::string& image_dir) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open image CSV: " << filepath << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string ts_str, filename;
        std::getline(ss, ts_str, ',');
        std::getline(ss, filename, ',');

        // Trim whitespace
        while (!filename.empty() && (filename.front() == ' ' || filename.front() == '\t'))
            filename.erase(filename.begin());

        ImageData img;
        img.timestamp = std::stoull(ts_str);
        img.filepath = image_dir + filename;
        out.push_back(img);
    }
    return true;
}

bool DataLoader::parse_groundtruth_csv(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open ground truth file: " << filepath << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        auto next = [&]() -> double {
            std::getline(ss, token, ',');
            return std::stod(token);
        };

        GroundTruthPose gt;
        std::getline(ss, token, ',');
        gt.timestamp = std::stoull(token);

        double px = next(), py = next(), pz = next();
        gt.position = Eigen::Vector3d(px, py, pz);

        // TUM-VI: qw, qx, qy, qz
        double qw = next(), qx = next(), qy = next(), qz = next();
        gt.orientation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();

        double vx = next(), vy = next(), vz = next();
        gt.velocity = Eigen::Vector3d(vx, vy, vz);

        double bwx = next(), bwy = next(), bwz = next();
        gt.gyro_bias = Eigen::Vector3d(bwx, bwy, bwz);

        double bax = next(), bay = next(), baz = next();
        gt.accel_bias = Eigen::Vector3d(bax, bay, baz);

        ground_truth_.push_back(gt);
    }
    return true;
}

bool DataLoader::parse_sensor_yaml(const std::string& filepath) {
    // Minimal YAML parser for TUM-VI sensor.yaml
    // Extracts: intrinsics [fu, fv, cu, cv], distortion_coefficients, resolution,
    // and T_BS (body-to-sensor extrinsic)
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open sensor YAML: " << filepath << "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Parse intrinsics line: "    intrinsics: [fu, fv, cu, cv]"
        if (line.find("intrinsics:") != std::string::npos &&
            line.find("[") != std::string::npos) {
            auto start = line.find('[');
            auto end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string vals = line.substr(start + 1, end - start - 1);
                std::istringstream ss(vals);
                std::string tok;
                std::vector<double> v;
                while (std::getline(ss, tok, ',')) {
                    v.push_back(std::stod(tok));
                }
                if (v.size() >= 4) {
                    camera_.fx = v[0];
                    camera_.fy = v[1];
                    camera_.cx = v[2];
                    camera_.cy = v[3];
                }
            }
        }

        // Parse distortion_coefficients
        if (line.find("distortion_coefficients:") != std::string::npos &&
            line.find("[") != std::string::npos) {
            auto start = line.find('[');
            auto end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string vals = line.substr(start + 1, end - start - 1);
                std::istringstream ss(vals);
                std::string tok;
                camera_.distortion.clear();
                while (std::getline(ss, tok, ',')) {
                    camera_.distortion.push_back(std::stod(tok));
                }
            }
        }

        // Parse resolution
        if (line.find("resolution:") != std::string::npos &&
            line.find("[") != std::string::npos) {
            auto start = line.find('[');
            auto end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string vals = line.substr(start + 1, end - start - 1);
                std::istringstream ss(vals);
                std::string tok;
                std::vector<int> v;
                while (std::getline(ss, tok, ',')) {
                    v.push_back(std::stoi(tok));
                }
                if (v.size() >= 2) {
                    camera_.width = v[0];
                    camera_.height = v[1];
                }
            }
        }

        // Parse T_BS (4x4 matrix stored row-major in YAML)
        // Format: "    data: [r00, r01, ..., r33]"
        if (line.find("data:") != std::string::npos &&
            line.find("[") != std::string::npos) {
            auto start = line.find('[');
            auto end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string vals = line.substr(start + 1, end - start - 1);
                std::istringstream ss(vals);
                std::string tok;
                std::vector<double> v;
                while (std::getline(ss, tok, ',')) {
                    v.push_back(std::stod(tok));
                }
                if (v.size() == 16) {
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c)
                            camera_.T_cam_imu(r, c) = v[r * 4 + c];
                }
            }
        }
    }
    return true;
}

} // namespace vio
