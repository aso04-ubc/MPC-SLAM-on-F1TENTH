#include "Utils/ParameterUtils.hpp"
#include "Core/Executor.hpp"
#include "Utils/TypeUtils.hpp"
#include "Providers/ComputeWrapper.hpp"

int main(int argc, char **argv) {
    // Initialize ROS 2, create the application node, and spin it until shutdown.
    rclcpp::init(argc, argv);

    // Node creation is split out so tests / integration code can reuse it.
    auto node = CreateApplicationNode();

    // Use a shared global executor to reduce spinning boilerplate.
    auto &executor = GetGlobalExecutor();

    executor.add_node(node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}
