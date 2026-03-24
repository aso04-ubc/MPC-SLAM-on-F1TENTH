#pragma once

#include "AppPCH.h"
#include "Core/SafetyInfo.hpp"

namespace AEB {
    class LidarSafetyInfoProvider : public ISafetyInfoProvider {
    public:
        LidarSafetyInfoProvider();

        virtual ~LidarSafetyInfoProvider() override = default;

        virtual void OnInit(rclcpp::Node *node) override;

        [[nodiscard]] virtual SafetyInfo GetSafetyInfo() const override;

    private:
        void OnLaserScan(sensor_msgs::msg::LaserScan::SharedPtr msg);

        void OnOdometry(nav_msgs::msg::Odometry::SharedPtr msg);

    private:
        rclcpp::Node *m_Node{nullptr};
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_ScanSubscription;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdomSubscription;
        rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr m_DriveSubscription;

        std::atomic<double> m_CurrentSpeed{0.0};
        std::atomic<double> m_MinDistance{std::numeric_limits<double>::infinity()};
        std::atomic<double> m_MinTTC{std::numeric_limits<double>::infinity()};
        std::atomic<double> m_CurrentSteering{0.0};
    };
} // namespace AEB
