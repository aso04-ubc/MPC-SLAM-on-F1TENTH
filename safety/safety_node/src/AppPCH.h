// AppPCH.h
// Common precompiled header for the drive_control_node.
// Contains frequently used system and ROS2 includes to speed up compilation
// and keep source files concise.

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <thread>
#include <atomic>
#include <map>
#include <unordered_set>

#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <dev_b7_interfaces/msg/drive_control_message.hpp>

#include <boost/smart_ptr/atomic_shared_ptr.hpp>
#include <boost/make_shared.hpp>

// for non-blocking console inp
#include <poll.h>
#include <unistd.h>