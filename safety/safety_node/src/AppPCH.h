// AppPCH.h
// Common precompiled header for the drive_control_node.
// Contains frequently used system and ROS2 includes to speed up compilation
// and keep source files concise.

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <thread>
#include <atomic>
#include <map>

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <dev_b7_interfaces/msg/drive_control_message.hpp>

#include <argparse/argparse.hpp>

// for non-blocking console inp
#include <poll.h>
#include <unistd.h>