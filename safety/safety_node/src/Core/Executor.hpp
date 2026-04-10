#pragma once

#include "AppPCH.h"

// Shared multi-threaded executor used by the application.
// Keeping it global avoids duplicating spin logic across nodes / unit tests.
inline rclcpp::executors::MultiThreadedExecutor& GetGlobalExecutor() {
    static rclcpp::executors::MultiThreadedExecutor executor{{}, std::min(4u, std::thread::hardware_concurrency())};
    return executor;
}