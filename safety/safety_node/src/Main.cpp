#include "Utils/ParameterUtils.hpp"
#include "Core/Executor.hpp"
#include "Utils/TypeUtils.hpp"
#include "Providers/ComputeWrapper.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = CreateApplicationNode();

    auto &executor = GetGlobalExecutor();

    executor.add_node(node);

    executor.spin();

    rclcpp::shutdown();
    return 0;
}
