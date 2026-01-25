#include "rclcpp/rclcpp.hpp"

#include "ControlSubmissionNode.hpp"

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<ControlSubmissionNode>();

    RCLCPP_INFO(node->get_logger(), "Control Submission Mux Node with integrated AEB has started.");
    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;
}