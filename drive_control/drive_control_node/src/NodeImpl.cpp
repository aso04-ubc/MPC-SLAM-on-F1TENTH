#include "Application.hpp"

namespace Impl {
    using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    class DriveControl : public ILifecycleNodeEXT {
    public:
        DriveControl(const NodeCreationInfo& info)
            : ILifecycleNodeEXT("drive_control_node"),
              TTCThreshold(info.aeb_ttc_threshold),
              DistanceThreshold(info.aeb_minimum_distance) {
            RCLCPP_INFO(get_logger(), "DriveControl Node Object Created.");
        }

        virtual ~DriveControl() {
            StopThreads();
        }


        CallbackReturn on_configure(const rclcpp_lifecycle::State&) override;

        CallbackReturn on_activate(const rclcpp_lifecycle::State&) override;

        CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override;

        CallbackReturn on_cleanup(const rclcpp_lifecycle::State&) override;

        CallbackReturn on_shutdown(const rclcpp_lifecycle::State&) override;

    private:
        void AEBLoop();

        void ConsoleLoop();

        void StopThreads();

        void OnProcessOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
            m_CurrentSpeed.store(msg->twist.twist.linear.x, std::memory_order_release);
        }

        void OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg);

        void OnDriveCommand(const dev_b7_interfaces::msg::DriveControlMessage::SharedPtr msg);

    private:
        const double TTCThreshold;
        const double DistanceThreshold;

    private:
        std::atomic<double> m_CurrentSpeed{0.0};
        std::atomic<bool> m_IsAEBActive{false};
        std::atomic<bool> m_ShouldStop{false};

    private:
        std::mutex m_MapAccessMutex;
        std::map<int32_t, dev_b7_interfaces::msg::DriveControlMessage::SharedPtr> mPriorityToLastMessageMap;

        std::timed_mutex m_LastDriveControlMessageMutex;
        ackermann_msgs::msg::AckermannDriveStamped m_LastReceivedMessage;

    private:
        std::thread m_AEBSubmissionThread;
        std::thread m_ConsoleInputListenerThread;

    private:
        rclcpp_lifecycle::LifecyclePublisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
        m_AckermannDrivePublisher;

        rclcpp::Subscription<dev_b7_interfaces::msg::DriveControlMessage>::SharedPtr m_DriveControlSubscription;
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_ScanSubscription;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdomSubscription;

    private:
        std::weak_ptr<IExecutionContext> m_CurrentExecutionContext;

    public:
        virtual void OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) override;

        virtual void OnDetachFromContext(const std::shared_ptr<IExecutionContext>& context) override;
    };

    CallbackReturn DriveControl::on_configure(const rclcpp_lifecycle::State& state) {
        RCLCPP_INFO(get_logger(), "Configuring: Initializing Subscriptions and Publishers...");

        m_AckermannDrivePublisher = this->create_publisher<
            ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);

        m_DriveControlSubscription = this->create_subscription<dev_b7_interfaces::msg::DriveControlMessage>(
            dev_b7_interfaces::msg::DriveControlMessage::BUILTIN_TOPIC_NAME_STRING, 10,
            std::bind(&DriveControl::OnDriveCommand, this, std::placeholders::_1));

        m_ScanSubscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&DriveControl::OnProcessLaserScan, this, std::placeholders::_1));

        m_OdomSubscription = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&DriveControl::OnProcessOdometry, this, std::placeholders::_1));

        return CallbackReturn::SUCCESS;
    }

    CallbackReturn DriveControl::on_activate(const rclcpp_lifecycle::State& state) {
        RCLCPP_INFO(get_logger(), "Activating: Starting Control Threads...");


        m_ShouldStop.store(false, std::memory_order_release);
        m_AEBSubmissionThread = std::thread(&DriveControl::AEBLoop, this);
        m_ConsoleInputListenerThread = std::thread(&DriveControl::ConsoleLoop, this);

        m_AckermannDrivePublisher->on_activate();

        return CallbackReturn::SUCCESS;
    }

    CallbackReturn DriveControl::on_deactivate(const rclcpp_lifecycle::State& state) {
        RCLCPP_INFO(get_logger(), "Deactivating: Stopping Threads...");

        m_AckermannDrivePublisher->on_deactivate();

        StopThreads();

        return CallbackReturn::SUCCESS;
    }

    CallbackReturn DriveControl::on_cleanup(const rclcpp_lifecycle::State& state) {
        RCLCPP_INFO(get_logger(), "Cleaning up resources...");
        m_AckermannDrivePublisher.reset();
        m_DriveControlSubscription.reset();
        m_ScanSubscription.reset();
        m_OdomSubscription.reset();
        return CallbackReturn::SUCCESS;
    }

    CallbackReturn DriveControl::on_shutdown(const rclcpp_lifecycle::State& state) {
        RCLCPP_INFO(get_logger(), "Shutting down node...");
        StopThreads();
        return CallbackReturn::SUCCESS;
    }

    void DriveControl::AEBLoop() {
        while (rclcpp::ok() && !m_ShouldStop.load(std::memory_order_acquire)) {
            if (m_IsAEBActive.load(std::memory_order_acquire)) {
                ackermann_msgs::msg::AckermannDriveStamped stop_msg;
                stop_msg.header.stamp = this->now();
                stop_msg.header.frame_id = "base_link";

                {
                    std::unique_lock lock(m_LastDriveControlMessageMutex, std::chrono::milliseconds(10));
                    if (lock.owns_lock()) {
                        stop_msg = m_LastReceivedMessage;
                    }
                }
                stop_msg.drive.speed = 0.0;
                m_AckermannDrivePublisher->publish(stop_msg);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    void DriveControl::ConsoleLoop() {
        pollfd fd{};
        fd.fd = STDIN_FILENO;
        fd.events = POLLIN;

        while (rclcpp::ok() && !m_ShouldStop.load(std::memory_order_acquire)) {
            int ret = poll(&fd, 1, 100);

            if (ret < 0) {
                if (errno == EINTR) continue;
                break;
            }

            if (ret > 0 && fd.revents & POLLIN) {
                std::string line;
                if (std::getline(std::cin, line)) {
                    if (line == "exit" || line == "quit") {
                        m_ShouldStop.store(true, std::memory_order_release);
                        if (auto executor = m_CurrentExecutionContext.lock()) {
                            executor->Detach(std::static_pointer_cast<ILifecycleNodeEXT>(shared_from_this()));
                        }
                        break;
                    } else if (line == "aeb_on") {
                        m_IsAEBActive.store(true, std::memory_order_release);
                        RCLCPP_WARN(get_logger(), "AEB Engaged manually.");
                    } else if (line == "aeb_off") {
                        m_IsAEBActive.store(false, std::memory_order_release);
                        RCLCPP_INFO(get_logger(), "AEB Released manually.");
                    } else if (!line.empty()) {
                        RCLCPP_INFO(get_logger(), "Unknown command: %s", line.c_str());
                    }
                }
            }
        }
        RCLCPP_INFO(get_logger(), "ConsoleLoop thread exiting.");
    }

    void DriveControl::StopThreads() {
        m_ShouldStop.store(true, std::memory_order_release);

        if (m_AEBSubmissionThread.joinable()) {
            m_AEBSubmissionThread.join();
        }

        if (m_ConsoleInputListenerThread.joinable()) {
            m_ConsoleInputListenerThread.join();
        }
    }

    void DriveControl::OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        double v = m_CurrentSpeed.load(std::memory_order_acquire);
        double min_ttc = std::numeric_limits<double>::infinity();
        double min_distance = std::numeric_limits<double>::infinity();

        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            double r = msg->ranges[i];

            if (std::isnan(r) || std::isinf(r) || r < msg->range_min || r > msg->range_max) {
                continue;
            }

            min_distance = std::min(min_distance, r);

            double angle = msg->angle_min + i * msg->angle_increment;
            double range_rate = v * std::cos(angle);

            if (range_rate > 0) {
                double ttc = r / range_rate;
                if (ttc < min_ttc) {
                    min_ttc = ttc;
                }
            }
        }

        if (min_ttc < TTCThreshold || min_distance < DistanceThreshold) {
            if (!m_IsAEBActive.load(std::memory_order_acquire)) {
                RCLCPP_ERROR(get_logger(), "AEB TRIGGERED! TTC: %.3f s, Dist: %.3f m", min_ttc, min_distance);
                m_IsAEBActive.store(true, std::memory_order_release);
            }
        } else {
            if (m_IsAEBActive.load(std::memory_order_acquire)) {
                m_IsAEBActive.store(false, std::memory_order_release);
                RCLCPP_INFO(get_logger(), "AEB Released.");
            }
        }
    }

    void DriveControl::OnDriveCommand(const dev_b7_interfaces::msg::DriveControlMessage::SharedPtr msg) {
        {
            std::lock_guard<std::mutex> lock(m_MapAccessMutex);
            if (msg->active) {
                mPriorityToLastMessageMap[msg->priority] = msg;
            } else {
                mPriorityToLastMessageMap.erase(msg->priority);
                goto handle_message_store;
            }

            if (auto highest_priority_msg(
                    mPriorityToLastMessageMap.rbegin()->second);
                highest_priority_msg && !m_IsAEBActive.load(std::memory_order_acquire)) {
                m_AckermannDrivePublisher->publish(highest_priority_msg->drive);
            }
        }

    handle_message_store: // I know this is ugly.

        {
            std::lock_guard<std::timed_mutex> lock(m_LastDriveControlMessageMutex);
            m_LastReceivedMessage = msg->drive;
        }
    }

    void DriveControl::OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) {
        m_CurrentExecutionContext = context;
    }

    void DriveControl::OnDetachFromContext(const std::shared_ptr<IExecutionContext>& context) {
        m_CurrentExecutionContext.reset();
    }
}

std::shared_ptr<ILifecycleNodeEXT> CreateApplicationNode(const NodeCreationInfo& creation_info) {
    return std::make_shared<Impl::DriveControl>(creation_info);
}

// void SetExecutorCurrent(
//     const std::shared_ptr<rclcpp_lifecycle::LifecycleNode>& node,
//     const std::shared_ptr<rclcpp::Executor>& executor) {
//     if (auto drive_control_node = std::dynamic_pointer_cast<Impl::DriveControl>(node)) {
//         drive_control_node->SetCurrentExecutor(executor);
//     }
// }
