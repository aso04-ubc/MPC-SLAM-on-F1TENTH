#include "Application.hpp"

namespace Impl {
    class DriveControl : public LifeTimeNode {
    public:
        virtual ~DriveControl() override = default;

        DriveControl(const NodeCreationInfo& info)
            : LifeTimeNode("drive_control_node"),
              TTCThreshold(info.aeb_ttc_threshold),
              MinimumDistance(info.aeb_minimum_distance) {
        }

    private:
        virtual void OnInit() override;
        virtual void OnDestroy() override;

    private:
        std::mutex m_MapAccessMutex{};
        std::map<
            int32_t,
            dev_b7_interfaces::msg::DriveControlMessage::SharedPtr
        > mPriorityToLastMessageMap{};

        // std::atomic_int32_t m_CurrentPriority{0};

        std::atomic_bool m_IsAEBActive{false};

        std::atomic_bool m_ShouldStop{false};

        std::thread m_AEBSubmissionThread;

    private:
        rclcpp::Subscription<dev_b7_interfaces::msg::DriveControlMessage>::SharedPtr m_DriveControlSubscription;
        rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr m_AckermannDrivePublisher;

    public:
        // AEB logic handler
        const double TTCThreshold;
        const double MinimumDistance;

    private:
        std::atomic<double> m_CurrentSpeed{0.0};

        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_ScanSubscription;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdomSubscription;

        // Integrate AEB directly in this node for lower latency
        void OnProcessOdometry(const nav_msgs::msg::Odometry::SharedPtr msg);
        void OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg);

    private:
        std::thread m_ConsoleInputListenerThread;

    private:
        std::mutex m_LastDriveControlMessageMutex;
        ackermann_msgs::msg::AckermannDriveStamped m_LastReceivedMessage;
    };

    void DriveControl::OnInit() {
        m_AEBSubmissionThread = std::thread{
            [self = std::weak_ptr(std::static_pointer_cast<DriveControl>(shared_from_this()))] {
                while (auto locked = self.lock()) {
                    if (locked->m_ShouldStop.load(std::memory_order_acquire)) {
                        break;
                    }

                    if (locked->m_IsAEBActive.load(std::memory_order_acquire)) {
                        // Publish zero speed command
                        ackermann_msgs::msg::AckermannDriveStamped stop_msg;
                        {
                            std::lock_guard lock(locked->m_LastDriveControlMessageMutex);
                            stop_msg = locked->m_LastReceivedMessage;
                        }
                        stop_msg.drive.speed = 0.0;
                        locked->m_AckermannDrivePublisher->publish(stop_msg);
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
            }
        };

        // Initialize subscriptions and publishers here
        m_DriveControlSubscription = this->create_subscription<dev_b7_interfaces::msg::DriveControlMessage>(
            dev_b7_interfaces::msg::DriveControlMessage::BUILTIN_TOPIC_NAME_STRING, 10,
            [this](const dev_b7_interfaces::msg::DriveControlMessage::SharedPtr msg) {
                // log the message
                RCLCPP_INFO(this->get_logger(), "Received DriveControlMessage with priority %d", msg->priority);
                {
                    std::lock_guard lock(m_MapAccessMutex);

                    if (msg->active) {
                        mPriorityToLastMessageMap[msg->priority] = msg;
                    } else {
                        mPriorityToLastMessageMap.erase(msg->priority);
                    }

                    if (m_IsAEBActive || mPriorityToLastMessageMap.empty()) {
                        return;
                    }

                    // Now we should publish the last message, if it is not active, it should already be erased
                    auto last = mPriorityToLastMessageMap.rbegin()->second;
                    if (last) {
                        m_AckermannDrivePublisher->publish(last->drive);
                    }
                }

                {
                    std::lock_guard lock(m_LastDriveControlMessageMutex);
                    m_LastReceivedMessage = msg->drive;
                }
            });

        m_AckermannDrivePublisher = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);

        m_ScanSubscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&DriveControl::OnProcessLaserScan, this, std::placeholders::_1));

        m_OdomSubscription = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10,
            std::bind(&DriveControl::OnProcessOdometry, this, std::placeholders::_1));

        m_ConsoleInputListenerThread =
            std::thread{
                [self = std::weak_ptr(std::static_pointer_cast<DriveControl>(shared_from_this()))] {
                    std::string line;
                    while (auto locked = self.lock()) {
                        std::getline(std::cin, line);
                        if (line == "exit" || line == "quit") {
                            locked->m_ShouldStop = true;
                            locked->Node::get_node_options()
                                  .context()
                                  ->shutdown("Shutdown requested by user command.");
                        } else if (line == "aeb_on") {
                            locked->m_IsAEBActive.store(true, std::memory_order_release);
                            RCLCPP_INFO(locked->get_logger(), "AEB Engaged by user command.");
                        } else if (line == "aeb_off") {
                            locked->m_IsAEBActive.store(false, std::memory_order_release);
                            RCLCPP_INFO(locked->get_logger(), "AEB Released by user command.");
                        } else {
                            RCLCPP_WARN(locked->get_logger(), "Unknown command %s", line.c_str());
                        }
                    }
                }
            };

        RCLCPP_INFO(this->get_logger(), "DriveControl Node Initialized.");
    }

    void DriveControl::OnDestroy() {
        m_ShouldStop.store(true, std::memory_order_release);

        m_ConsoleInputListenerThread.detach(); // Detach since it may be blocked on std::getline,
        // we don't care about the resource leak anyways because we are terminating.
    }

    void DriveControl::OnProcessOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
        m_CurrentSpeed.store(msg->twist.twist.linear.x, std::memory_order_release);
    }

    void DriveControl::OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
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

        if (min_ttc < TTCThreshold) {
            if (!m_IsAEBActive.load(std::memory_order_acquire)) {
                RCLCPP_WARN(this->get_logger(), "AEB Triggered! TTC: %.3f", min_ttc);
                m_IsAEBActive.store(true, std::memory_order_release);
            }
        } else if (min_distance > MinimumDistance) {
            m_IsAEBActive.store(false, std::memory_order_release);
        }
    }
}

std::shared_ptr<LifeTimeNode> CreateApplicationNode(const NodeCreationInfo& creation_info) {
    return std::make_shared<Impl::DriveControl>(creation_info);
}
