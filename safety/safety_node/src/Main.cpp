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

    // Create configuration structure with parsed values
    NodeCreationInfo info;
    info.aeb_ttc_threshold = 0.5;
    info.aeb_minimum_distance = 0.25;

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
