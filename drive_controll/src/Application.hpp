#pragma once

#include "AppPCH.h"

struct NodeCreationInfo {
    double aeb_ttc_threshold = 0.8;
    double aeb_minimum_distance = 0.3;
};

class LifeTimeNode : public rclcpp::Node {
public:
    using Node::Node;

    virtual void OnInit() = 0;
    virtual void OnDestroy() = 0;
};

std::shared_ptr<LifeTimeNode> CreateApplicationNode(const NodeCreationInfo& creation_info);

