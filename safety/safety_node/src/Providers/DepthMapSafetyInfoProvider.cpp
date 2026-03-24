#include "Providers/DepthMapSafetyInfoProvider.hpp"

#include "Utils/ParameterUtils.hpp"
#include "Utils/TypeUtils.hpp"

namespace AEB {
    DepthMapSafetyInfoProvider::DepthMapSafetyInfoProvider(ComputeMode computeMode) : m_ComputeMode(computeMode) {
    }

    void DepthMapSafetyInfoProvider::OnInit(rclcpp::Node *node) {
        m_Node = node;
        m_DepthMapToBucketComputer = IDepthMapToBucketComputer::Create(m_ComputeMode);

        RegionOfInterestInfo regionOfInterestInfo{};
        regionOfInterestInfo.StartX = NodeUtils::GetParameter<uint32_t>(node, "depth_map_roi_start_x").value_or(100);
        regionOfInterestInfo.EndX = NodeUtils::GetParameter<uint32_t>(node, "depth_map_roi_end_x").value_or(540);
        regionOfInterestInfo.StartY = NodeUtils::GetParameter<uint32_t>(node, "depth_map_roi_start_y").value_or(280);
        regionOfInterestInfo.EndY = NodeUtils::GetParameter<uint32_t>(node, "depth_map_roi_end_y").value_or(360);

        LowestProportionFromDepthMapComputeInfo::ConfigType config{};
        config.Proportion = NodeUtils::GetParameter<float>(node, "depth_map_proportion").value_or(0.01f);
        config.DistanceMinInMillimeter = NodeUtils::GetParameter<uint32_t>(node, "depth_map_distance_min_in_mm").
                value_or(200);
        config.DistanceMaxInMillimeter = NodeUtils::GetParameter<uint32_t>(node, "depth_map_distance_max_in_mm").
                value_or(4000);
        config.StepSizeInMillimeter = NodeUtils::GetParameter<uint32_t>(node, "depth_map_step_size_in_mm").value_or(10);

        m_ComputeInfo.RegionOfInterest = regionOfInterestInfo;
        m_ComputeInfo.Config = config;

        m_ScratchBuckets.reserve(
            (config.DistanceMaxInMillimeter - config.DistanceMinInMillimeter) / config.StepSizeInMillimeter + 1);

        m_DepthMapSubscription = node->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_rect_raw",
            rclcpp::SensorDataQoS(),
            std::bind(&DepthMapSafetyInfoProvider::OnDepthMap, this, std::placeholders::_1)
        );

        bool isSim = NodeUtils::GetParameter<bool>(node, "sim").value_or(false);
        std::string odom_topic = isSim ? "/ego_racecar/odom" : "/odom";
        m_OdomSubscription = node->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic,
            rclcpp::SensorDataQoS(),
            std::bind(&DepthMapSafetyInfoProvider::OnOdometry, this, std::placeholders::_1)
        );
    }

    SafetyInfo DepthMapSafetyInfoProvider::GetSafetyInfo() const {
        SafetyInfo info;
        info.MinDistance = m_MinDistance.load(std::memory_order_acquire);
        info.MinTTC = m_MinTTC.load(std::memory_order_acquire);
        info.Source = "DepthMap";
        info.Timestamp = m_Node->now();
        return info;
    }

    void DepthMapSafetyInfoProvider::OnDepthMap(sensor_msgs::msg::Image::SharedPtr msg) {
        static std::atomic_size_t m_SuboptimalWarningCount{0};
        static std::atomic_size_t m_TotalCount{0};

        m_TotalCount.fetch_add(1, std::memory_order_relaxed);

        DepthMapInfo depthMapInfo;

        assert(msg->encoding == "16UC1" && "Expected depth map encoding to be 16UC1");
        assert(msg->is_bigendian == 0 && "Expected depth map to be little-endian");

        depthMapInfo.Buffer = reinterpret_cast<const uint16_t *>(msg->data.data());
        depthMapInfo.Width = msg->width;
        depthMapInfo.Height = msg->height;

        m_ComputeInfo.DepthMapInMillimeter = depthMapInfo;

        m_ScratchBuckets.clear();

        auto result = GetLowestProportionFromDepthMap(*m_DepthMapToBucketComputer, m_ScratchBuckets, m_ComputeInfo);

        if (result) {
            uint32_t lowestProportionDistance = *result;
            m_MinDistance.store(static_cast<double>(lowestProportionDistance) / 1000.0, std::memory_order_release);

            double speed = m_CurrentSpeed.load(std::memory_order_acquire);
            if (speed > 0) {
                double ttc = static_cast<double>(lowestProportionDistance) / 1000.0 / speed;
                m_MinTTC.store(ttc, std::memory_order_release);
            } else {
                m_MinTTC.store(std::numeric_limits<double>::infinity(), std::memory_order_release);
            }
        } else {
            m_SuboptimalWarningCount.fetch_add(1, std::memory_order_relaxed);
            RCLCPP_WARN(m_Node->get_logger(), "Failed to compute safety info from depth map");
        }
    }

    void DepthMapSafetyInfoProvider::OnOdometry(nav_msgs::msg::Odometry::SharedPtr msg) {
        m_CurrentSpeed.store(std::abs(msg->twist.twist.linear.x), std::memory_order_release);
    }
}
