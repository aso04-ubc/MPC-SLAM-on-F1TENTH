//
// Created by yiran on 1/22/26.
//

#ifndef CONTROL_SUBMISSION_CONTROLSUBMISSIONNODE_HPP
#define CONTROL_SUBMISSION_CONTROLSUBMISSIONNODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "dev_b7_interfaces/msg/control_submission_message.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include <mutex>
#include <atomic>
#include <memory>
#include <queue>
#include <vector>
#include <condition_variable>
#include <thread>

namespace Impl {
    struct MyMessageComp {
        bool operator()(const dev_b7_interfaces::msg::ControlSubmissionMessage::SharedPtr& lhs,
                        const dev_b7_interfaces::msg::ControlSubmissionMessage::SharedPtr& rhs) const {
            return lhs->priority < rhs->priority; // Item with higher priority should be popped first
        }
    };
}

class ControlSubmissionNode : public rclcpp::Node {
public:
    ControlSubmissionNode() : Node("control_submission_node") {
        OnInit();
    }

    virtual ~ControlSubmissionNode() override {
        OnDestroy();
    }

private:
    void OnInit();

    void OnDestroy();

private:
    std::mutex m_QueueAccessMutex{};
    std::priority_queue<
        dev_b7_interfaces::msg::ControlSubmissionMessage::SharedPtr,
        std::vector<dev_b7_interfaces::msg::ControlSubmissionMessage::SharedPtr>,
        Impl::MyMessageComp> m_MessageQueue{};
    std::atomic_bool m_IsAEBActive{false};

    std::atomic_bool m_ShouldStop{false};

    std::condition_variable m_ConditionVariable{};

    std::thread m_SubmitterThread;

private:
    rclcpp::Subscription<dev_b7_interfaces::msg::ControlSubmissionMessage>::SharedPtr m_ControlSubmissionSubscription;
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr m_AckermannDrivePublisher;

private:
    // AEB logic handler
    constexpr static double s_TTCThreshold = 0.8;
    constexpr static double s_MinimumSpeed = 0.1;
    std::atomic<double> m_CurrentSpeed{0.0};

    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_ScanSubscription;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdomSubscription;

    // Integrate AEB directly in this node for lower latency
    void OnProcessOdometry(const nav_msgs::msg::Odometry::SharedPtr msg);
    void OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg);
};

#endif //CONTROL_SUBMISSION_CONTROLSUBMISSIONNODE_HPP