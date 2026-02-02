/**
 * @file Main.cpp
 * @brief Entry point for the Drive Control Node
 *
 * This file contains the main() function that:
 * 1. Initializes ROS2
 * 2. Parses command-line arguments for AEB configuration
 * 3. Creates and configures the drive control node
 * 4. Sets up a multi-threaded execution context
 * 5. Runs the node until shutdown
 */

#include "Application.hpp"

/**
 * @brief Main entry point for the Drive Control Node
 *
 * EXECUTION FLOW:
 * ===============
 * 1. Initialize ROS2 with command-line arguments
 * 2. Parse non-ROS arguments using argparse library
 *    - --aeb-ttc-threshold (-t): Time-to-collision threshold (default: 0.3s)
 *    - --aeb-minimum-distance (-d): Minimum distance threshold (default: 0.5m)
 * 3. Create the drive control node with parsed configuration
 * 4. Transition node through lifecycle states:
 *    - Configure: Initialize publishers and subscribers
 *    - Activate: Start AEB and console threads
 * 5. Spin the executor to process callbacks
 * 6. On shutdown:
 *    - Deactivate: Stop threads
 *    - Cleanup: Release resources
 * 7. Shutdown ROS2
 *
 * @param argc Argument count
 * @param argv Argument vector
 * @return int Exit code (0 for success, -1 for failure)
 */
int main(int argc, char** argv) {
    // Initialize ROS2 middleware
    rclcpp::init(argc, argv);

    // Separate ROS2 arguments from application-specific arguments
    // ROS2 arguments (e.g., --ros-args) are removed, leaving only our custom args
    std::vector<std::string> non_ros_args_strings = rclcpp::remove_ros_arguments(argc, argv);

    // Convert std::vector<std::string> to char** for argparse compatibility
    std::vector<char*> new_argv_vec;
    for (const auto& arg : non_ros_args_strings) {
        new_argv_vec.push_back(const_cast<char*>(arg.c_str()));
    }
    new_argv_vec.push_back(nullptr); // Null-terminate the argument list

    int new_argc = static_cast<int>(non_ros_args_strings.size());
    char** new_argv = new_argv_vec.data();

    // Set up argument parser for AEB configuration
    argparse::ArgumentParser parser("Drive Control Node");

    // Define command-line argument: Time-To-Collision threshold
    // This determines how much time before collision the AEB should engage
    parser.add_argument("--aeb-ttc-threshold", "-t")
          .help("Time-To-Collision threshold for AEB activation (in seconds)")
          .default_value(0.3)
          .scan<'g', double>();

    // Define command-line argument: Minimum distance threshold
    // This is the absolute minimum distance before AEB engages, regardless of TTC
    parser.add_argument("--aeb-minimum-distance", "-d")
          .help("Minimum distance threshold for AEB deactivation (in meters)")
          .default_value(0.3)
          .scan<'g', double>();

    // Parse the arguments and handle errors
    try {
        parser.parse_args(new_argc, new_argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser.help().str() << std::endl;
        return -1;
    }

    // Create configuration structure with parsed values
    NodeCreationInfo info;
    info.aeb_ttc_threshold = parser.get<double>("--aeb-ttc-threshold");
    info.aeb_minimum_distance = parser.get<double>("--aeb-minimum-distance");

    // Create the drive control node using factory function
    auto node = CreateApplicationNode(info);

    // Transition to CONFIGURED state: initializes publishers/subscribers
    node->configure();

    // Create a multi-threaded executor with 4 worker threads
    // This allows parallel processing of callbacks for better performance
    auto executionContext = std::make_shared<ExecutionContextMultithreaded>(4);

    // Attach the node to the executor
    executionContext->Attach(node);

    // Transition to ACTIVE state: starts AEB loop and console input threads
    node->activate();

    // Run the executor (blocking call until shutdown)
    // Processes incoming messages and triggers callbacks
    executionContext->Spin();

    // Transition to INACTIVE state: stops threads gracefully
    node->deactivate();

    // Transition to UNCONFIGURED state: releases resources
    node->cleanup();

    // Shutdown ROS2 middleware
    rclcpp::shutdown();
    return 0;
}
