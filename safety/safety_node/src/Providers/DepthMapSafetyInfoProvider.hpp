#pragma once

#include "AppPCH.h"
#include "Core/SafetyInfo.hpp"
#include "Providers/ComputeWrapper.hpp"

namespace AEB {
    class DepthMapSafetyInfoProvider : public ISafetyInfoProvider {
    public:
        DepthMapSafetyInfoProvider(ComputeMode computeMode);

        virtual void OnInit(rclcpp::Node *node) override;

        virtual SafetyInfo GetSafetyInfo() const override;

        virtual ~DepthMapSafetyInfoProvider() override = default;

    private:
        void OnDepthMap(sensor_msgs::msg::Image::SharedPtr msg);

        void OnOdometry(nav_msgs::msg::Odometry::SharedPtr msg);

    private:
        ComputeMode m_ComputeMode;

    private:
        rclcpp::Node *m_Node{};
        // Compute description for turning a depth map into a histogram-based "lowest proportion" distance.
        LowestProportionFromDepthMapComputeInfo m_ComputeInfo{};
        std::unique_ptr<IDepthMapToBucketComputer> m_DepthMapToBucketComputer;
        // Reusable histogram bucket buffer (each bucket stores pixel counts).
        std::vector<uint32_t> m_ScratchBuckets;

    private:
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_DepthMapSubscription;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdomSubscription;

    private:
        // Current longitudinal speed estimate (m/s) from odometry.
        std::atomic<double> m_CurrentSpeed{0.0};
        // Minimum estimated distance among the selected depth-map pixels (meters).
        std::atomic<double> m_MinDistance{std::numeric_limits<double>::infinity()};
        // Minimum estimated TTC based on distance/speed (seconds).
        std::atomic<double> m_MinTTC{std::numeric_limits<double>::infinity()};
    };
}
