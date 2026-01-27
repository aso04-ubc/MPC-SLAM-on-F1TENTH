#include "Application.hpp"
#include <vector>
#include <string>
#include <algorithm>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::vector<std::string> non_ros_args_strings = rclcpp::remove_ros_arguments(argc, argv);

    std::vector<char*> new_argv_vec;
    for (const auto& arg : non_ros_args_strings) {
        new_argv_vec.push_back(const_cast<char*>(arg.c_str()));
    }
    new_argv_vec.push_back(nullptr);

    int new_argc = static_cast<int>(non_ros_args_strings.size());
    char** new_argv = new_argv_vec.data();

    argparse::ArgumentParser parser("Drive Control Node");

    parser.add_argument("--aeb-ttc-threshold", "-t")
          .help("Time-To-Collision threshold for AEB activation (in seconds)")
          .default_value(0.3)
          .scan<'g', double>();

    parser.add_argument("--aeb-minimum-distance", "-d")
          .help("Minimum distance threshold for AEB deactivation (in meters)")
          .default_value(0.5)
          .scan<'g', double>();

    try {
        parser.parse_args(new_argc, new_argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser.help().str() << std::endl;
        return -1;
    }

    NodeCreationInfo info;
    info.aeb_ttc_threshold = parser.get<double>("--aeb-ttc-threshold");
    info.aeb_minimum_distance = parser.get<double>("--aeb-minimum-distance");

    auto node = CreateApplicationNode(info);
    node->configure();

    auto executionContext = std::make_shared<ExecutionContextMultithreaded>(4);

    executionContext->Attach(node);
    node->activate();
    executionContext->Spin();
    node->deactivate();

    node->cleanup();

    rclcpp::shutdown();
    return 0;
}
