#include "Application.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

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
        parser.parse_args(argc, argv);
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
