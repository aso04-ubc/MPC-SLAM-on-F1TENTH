#include "Application.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    argparse::ArgumentParser parser("Drive Control Node");

    parser.add_argument("--aeb-ttc-threshold", "-t")
          .help("Time-To-Collision threshold for AEB activation (in seconds)")
          .default_value(0.2)
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

    auto node = CreateApplicationNode(
        NodeCreationInfo{
            parser.get<double>("--aeb-ttc-threshold"),
            parser.get<double>("--aeb-minimum-distance"),
        });

    node->OnInit();

    rclcpp::ExecutorOptions options;
    auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(options, 4);
    executor->add_node(node);
    executor->spin();
    executor->remove_node(node);
    executor.reset();

    node->OnDestroy();

    rclcpp::shutdown();
    return 0;
}
