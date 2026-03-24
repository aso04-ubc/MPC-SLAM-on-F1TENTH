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
        LowestProportionFromDepthMapComputeInfo m_ComputeInfo{};
        std::unique_ptr<IDepthMapToBucketComputer> m_DepthMapToBucketComputer;
        std::vector<uint32_t> m_ScratchBuckets; // Reusable buffer for histogram buckets

    private:
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr m_DepthMapSubscription;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdomSubscription;

    private:
        std::atomic<double> m_CurrentSpeed{0.0};
        std::atomic<double> m_MinDistance{std::numeric_limits<double>::infinity()};
        std::atomic<double> m_MinTTC{std::numeric_limits<double>::infinity()};
    };
}
