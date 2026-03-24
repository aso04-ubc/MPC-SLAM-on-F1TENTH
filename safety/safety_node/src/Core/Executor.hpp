#pragma once

#include "AppPCH.h"

inline rclcpp::executors::MultiThreadedExecutor& GetGlobalExecutor() {
    static rclcpp::executors::MultiThreadedExecutor executor{{}, std::min(4u, std::thread::hardware_concurrency())};
    return executor;
}