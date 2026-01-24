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

    m_SubmitterThread = std::thread([this] {
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

    m_ConsoleInputListenerThread = std::thread([this] {
        while (!m_ShouldStop.load(std::memory_order_acquire)) {
            std::string input;
            std::getline(std::cin, input);
            if (input == "exit" || input == "quit") {
                m_ShouldStop.store(true, std::memory_order_release);
                m_ConditionVariable.notify_all();
                RCLCPP_INFO(this->get_logger(), "Shutdown requested by user command.");
                Node::get_node_options()
                    .context()->shutdown("User requested shutdown.");
            } else if (input == "release_aeb") {
                m_IsAEBActive.store(false, std::memory_order_release);
                RCLCPP_INFO(this->get_logger(), "AEB Released by user command.");
            } else {
                RCLCPP_WARN(this->get_logger(), "Unknown command %s", input.c_str());
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

    m_ConsoleInputListenerThread.detach(); // Detach since it may be blocked on std::getline,
    // we don't care about the resource leak anyways because we are terminating.
}

void ControlSubmissionNode::OnProcessOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
    m_CurrentSpeed.store(msg->twist.twist.linear.x, std::memory_order_release);
}

void ControlSubmissionNode::OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    double v = m_CurrentSpeed.load(std::memory_order_acquire);

    double min_ttc = std::numeric_limits<double>::infinity();

    double min_distance = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < msg->ranges.size(); ++i) {
        double r = msg->ranges[i];

        min_distance = std::min(min_distance, r);

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
        if (!m_IsAEBActive.load(std::memory_order_acquire)) {
            RCLCPP_WARN(this->get_logger(), "AEB Triggered! TTC: %.3f", min_ttc);
            m_IsAEBActive.store(true, std::memory_order_release);
            m_ConditionVariable.notify_one();
        }
    } else if (min_distance > s_MinimumDistance) {
        m_IsAEBActive.store(false, std::memory_order_release);
    }
}
