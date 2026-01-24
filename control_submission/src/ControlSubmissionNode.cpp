#include "ControlSubmissionNode.hpp"

void ControlSubmissionNode::OnInit() {
    m_ControlSubmissionSubscription = this->create_subscription<dev_b7_interfaces::msg::ControlSubmissionMessage>(
        "/control_submission", 10,
        [this](dev_b7_interfaces::msg::ControlSubmissionMessage::SharedPtr msg) {
            {
                std::lock_guard<std::mutex> lock(m_QueueAccessMutex);
                m_MessageQueue.push(msg);
            }
            m_ConditionVariable.notify_one();
        });


    m_OdomSubscription = this->create_subscription<nav_msgs::msg::Odometry>(
        "/ego_racecar/odom", 10, [this](nav_msgs::msg::Odometry::SharedPtr msg) {
            OnProcessOdometry(msg);
        });

    m_ScanSubscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10, [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {
            OnProcessLaserScan(msg);
        });

    m_AckermannDrivePublisher = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
        "/drive", 10);

    m_SubmitterThread = std::thread([this]() {
        while (!m_ShouldStop.load(std::memory_order_acquire)) {
            dev_b7_interfaces::msg::ControlSubmissionMessage::SharedPtr msg;

            {
                std::unique_lock<std::mutex> lock(m_QueueAccessMutex);
                m_ConditionVariable.wait_for(lock, std::chrono::milliseconds(20), [this] {
                    return !m_MessageQueue.empty() || m_ShouldStop.load() || m_IsAEBActive.load();
                });

                if (m_ShouldStop.load()) break;

                if (m_IsAEBActive.load(std::memory_order_acquire)) {
                    ackermann_msgs::msg::AckermannDriveStamped stop_msg;
                    stop_msg.drive.speed = 0.0;
                    stop_msg.drive.acceleration = 0.0;
                    m_AckermannDrivePublisher->publish(stop_msg);

                    while (!m_MessageQueue.empty()) m_MessageQueue.pop();
                    continue;
                }

                if (!m_MessageQueue.empty()) {
                    msg = std::move(m_MessageQueue.top());
                    m_MessageQueue.pop();
                }
            }

            if (msg) {
                m_AckermannDrivePublisher->publish(msg->drive);
            }
        }
    });
}

void ControlSubmissionNode::OnDestroy() {
    m_ShouldStop.store(true, std::memory_order_release);
    m_ConditionVariable.notify_all();

    if (m_SubmitterThread.joinable()) {
        m_SubmitterThread.join();
    }
}

void ControlSubmissionNode::OnProcessOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
    m_CurrentSpeed.store(msg->twist.twist.linear.x, std::memory_order_release);
}

void ControlSubmissionNode::OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    double v = m_CurrentSpeed.load(std::memory_order_acquire);
    if (v < s_MinimumSpeed) {
        m_IsAEBActive.store(false);
        return;
    }

    double min_ttc = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < msg->ranges.size(); ++i) {
        double r = msg->ranges[i];
        if (std::isnan(r) || std::isinf(r) || r < msg->range_min || r > msg->range_max) {
            continue;
        }

        double angle = msg->angle_min + i * msg->angle_increment;
        double range_rate = v * std::cos(angle);

        if (range_rate > 0) {
            double ttc = r / range_rate;
            if (ttc < min_ttc) {
                min_ttc = ttc;
            }
        }
    }

    if (min_ttc < s_TTCThreshold) {
        if (!m_IsAEBActive.load()) {
            RCLCPP_WARN(this->get_logger(), "AEB Triggered! TTC: %.3f", min_ttc);
            m_IsAEBActive.store(true);
            m_ConditionVariable.notify_one();
        }
    }
    else {
        m_IsAEBActive.store(false); // Could be a bad idea? Let's see if it works.
    }
}
