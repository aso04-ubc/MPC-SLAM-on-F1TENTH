#include "Application.hpp"

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);

    argparse::ArgumentParser parser("Drive Control Node");

    double aeb_ttc_threshold = 0.8;
    double aeb_minimum_distance = 0.3;

    parser.add_argument("--aeb_ttc_threshold")
        .help("Time-To-Collision threshold for AEB activation (in seconds). Default: 0.8")
        .default_value(std::to_string(aeb_ttc_threshold))
        .action([&aeb_ttc_threshold](const std::string& value) {
            aeb_ttc_threshold = std::stod(value);
        });

    parser.add_argument("--aeb_minimum_distance")
        .help("Minimum distance threshold for AEB activation (in meters). Default: 0.3")
        .default_value(std::to_string(aeb_minimum_distance))
        .action([&aeb_minimum_distance](const std::string& value) {
            aeb_minimum_distance = std::stod(value);
        });

    auto node = CreateApplicationNode(NodeCreationInfo{
        .aeb_ttc_threshold = aeb_ttc_threshold,
        .aeb_minimum_distance = aeb_minimum_distance
    });

    node->OnInit();

    rclcpp::spin(node);

    node->OnDestroy();

    rclcpp::shutdown();
    return 0;
}