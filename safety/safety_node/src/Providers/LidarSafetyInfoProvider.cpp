#include "Providers/LidarSafetyInfoProvider.hpp"

#include "Utils/ParameterUtils.hpp"
#include "Utils/TypeUtils.hpp"

namespace AEB {
    LidarSafetyInfoProvider::LidarSafetyInfoProvider() = default;

    void LidarSafetyInfoProvider::OnInit(rclcpp::Node *node) {
        m_Node = node;

        bool isSim = NodeUtils::GetParameter<bool>(node, "sim").value_or(false);
        std::string odom_topic = isSim ? "/ego_racecar/odom" : "/odom";

        m_ScanSubscription = node->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan",
            rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {
                OnLaserScan(msg);
            }
        );

        m_OdomSubscription = node->create_subscription<nav_msgs::msg::Odometry>(
            odom_topic,
            rclcpp::SensorDataQoS(),
            [this](nav_msgs::msg::Odometry::SharedPtr msg) {
                OnOdometry(msg);
            }
        );

        m_DriveSubscription = node->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
        "/drive", 10,
        [this](ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
            m_CurrentSteering.store(msg->drive.steering_angle, std::memory_order_release);
        }
);

        RCLCPP_INFO(node->get_logger(), "LidarAEBProvider initialized (odom: %s)", odom_topic.c_str());
    }

    void LidarSafetyInfoProvider::OnOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double speed = std::abs(msg->twist.twist.linear.x);
        m_CurrentSpeed.store(speed, std::memory_order_release);
    }

    void LidarSafetyInfoProvider::OnLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        double v = m_CurrentSpeed.load(std::memory_order_acquire);
        double current_steer = m_CurrentSteering.load(std::memory_order_acquire);

        double min_distance = std::numeric_limits<double>::infinity();
        double min_ttc = std::numeric_limits<double>::infinity();

        const double angle_tolerance = 0.4;
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            double r = msg->ranges[i];

            if (std::isnan(r) || std::isinf(r) || r < msg->range_min || r > msg->range_max) {
                continue;
            }

            double angle = msg->angle_min + static_cast<double>(i) * msg->angle_increment;
            if (std::abs(angle-current_steer) > angle_tolerance) continue;

            min_distance = std::min(min_distance, r);

            double range_rate = v * std::cos(angle);

            if (range_rate > 1e-3) {
                double ttc = r / range_rate;
                min_ttc = std::min(min_ttc, ttc);
            }
        }

        m_MinDistance.store(min_distance, std::memory_order_release);
        m_MinTTC.store(min_ttc, std::memory_order_release);
    }


    SafetyInfo LidarSafetyInfoProvider::GetSafetyInfo() const {
        SafetyInfo info;
        info.Source = "lidar";

        info.MinDistance = m_MinDistance.load(std::memory_order_acquire);
        info.MinTTC = m_MinTTC.load(std::memory_order_acquire);
        info.Timestamp = m_Node->now();

        return info;
    }
} // namespace AEB

std::shared_ptr<ISafetyInfoProvider> CreateLidarAEBProvider() {
    return std::make_shared<AEB::LidarSafetyInfoProvider>();
}
