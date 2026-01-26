#pragma once

#include "AppPCH.h"

struct NodeCreationInfo {
    double aeb_ttc_threshold = 0.8;
    double aeb_minimum_distance = 0.3;
};

std::shared_ptr<rclcpp_lifecycle::LifecycleNode> CreateApplicationNode(const NodeCreationInfo& creation_info);

void SetExecutorCurrent(
    const std::shared_ptr<rclcpp_lifecycle::LifecycleNode>& node,
    const std::shared_ptr<rclcpp::Executor>& executor);