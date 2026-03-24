#pragma once

#include "AppPCH.h"

enum class SafetyState : uint8_t {
    eOptimal = 0,
    eSuboptimal = 1,
    eCritical = 2,
    eEmergency = 3
};

struct SafetyInfo {
    double MinDistance;
    double MinTTC;
    std::string Source;
    rclcpp::Time Timestamp;

    SafetyInfo()
        : MinDistance(std::numeric_limits<double>::infinity()),
          MinTTC(std::numeric_limits<double>::infinity()),
          Source("unknown"),
          Timestamp(0, 0, RCL_ROS_TIME) {
    }
};

class ISafetyInfoProvider {
public:
    virtual ~ISafetyInfoProvider() = default;

    virtual void OnInit(rclcpp::Node* node) = 0;

    [[nodiscard]] virtual SafetyInfo GetSafetyInfo() const = 0;
};

std::shared_ptr<ISafetyInfoProvider> CreateLidarAEBProvider();
