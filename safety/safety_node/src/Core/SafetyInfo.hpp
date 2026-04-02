#pragma once

#include "AppPCH.h"

enum class SafetyState : uint8_t {
    eOptimal = 0,
    eSuboptimal = 1,
    eCritical = 2,
    eEmergency = 3
};

// Fused/derived safety metrics from one sensing provider (or after fusion).
// Units:
// - MinDistance: meters (m)
// - MinTTC: seconds (s)
struct SafetyInfo {
    double MinDistance;
    double MinTTC;
    // Human-readable provider / fusion source identifier (e.g., "lidar", "DepthMap", "fusion").
    std::string Source;
    // Time at which these metrics were produced / sampled.
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

    // Called once during node initialization to set up subscriptions/resources.
    virtual void OnInit(rclcpp::Node* node) = 0;

    // Returns the latest safety metrics.
    // Implementations typically update internal atomics in subscription callbacks.
    [[nodiscard]] virtual SafetyInfo GetSafetyInfo() const = 0;
};

std::shared_ptr<ISafetyInfoProvider> CreateLidarAEBProvider();
