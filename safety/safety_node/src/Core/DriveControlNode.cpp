#include "Utils/ParameterUtils.hpp"
#include "Core/Executor.hpp"
#include "Core/SafetyInfo.hpp"
#include "Providers/LidarSafetyInfoProvider.hpp"
#include "Providers/DepthMapSafetyInfoProvider.hpp"
#include "Core/DynamicParameter.hpp"
#include "Core/NodeBase.hpp"
#include <csignal>

namespace Impl {
    // A modifier transforms an AckermannDrive command depending on the active safety stage.
    // It is applied to the latest/highest-priority command before publishing to the drive topic.
    using MessageModifier = std::function<ackermann_msgs::msg::AckermannDrive(ackermann_msgs::msg::AckermannDrive &&)>;
    inline const static MessageModifier Identity = [](
        ackermann_msgs::msg::AckermannDrive &&msg) -> ackermann_msgs::msg::AckermannDrive {
        return msg;
    };

    class DriveControl : public INodeExtend {
    public:
        DriveControl() : INodeExtend("drive_control_node", {},
                                     rclcpp::NodeOptions().allow_undeclared_parameters(true)) {
            RCLCPP_INFO(get_logger(), "DriveControl Node Object Created.");

            m_AEBParameters.TTCWarning =
                    std::make_unique<DynamicParameter<double>>(this, "ttc_warning", 1.2);
            m_AEBParameters.TTCPartial =
                    std::make_unique<DynamicParameter<double>>(this, "ttc_partial", 0.9);
            m_AEBParameters.TTCFull =
                    std::make_unique<DynamicParameter<double>>(this, "ttc_full", 0.55);

            m_AEBParameters.DistanceWarning =
                    std::make_unique<DynamicParameter<double>>(this, "distance_warning", 0.7);
            m_AEBParameters.DistancePartial =
                    std::make_unique<DynamicParameter<double>>(this, "distance_partial", 0.6);
            m_AEBParameters.DistanceFull =
                    std::make_unique<DynamicParameter<double>>(this, "distance_full", 0.25);

            // m_AEBParameters.PartialBrakeDecel =
            //         std::make_unique<DynamicParameter<double>>(this, "partial_brake_decel", 2.0);
            // m_AEBParameters.PartialSpeedLimit =
            //         std::make_unique<DynamicParameter<double>>(this, "partial_speed_limit", 2.0);

            bool isSim = NodeUtils::GetParameter<bool>(this, "sim").value_or(false);

            m_AEBParameters.AutoRelease =
                    std::make_unique<DynamicParameter<bool>>(this, "aeb_auto_release", isSim);
        }

        virtual ~DriveControl() override {
            RCLCPP_INFO(get_logger(), "Shutting down node... Sending Emergency Stop.");
            
            ackermann_msgs::msg::AckermannDriveStamped stop_msg;
            stop_msg.header.stamp = this->now();
            stop_msg.header.frame_id = "base_link";
            stop_msg.drive.speed = 0.0;
            stop_msg.drive.acceleration = 10.0; // High braking force
            stop_msg.drive.steering_angle = 0.0;

            if (m_AckermannDrivePublisher) {
                m_AckermannDrivePublisher->publish(stop_msg);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            StopThreads();
            m_AckermannDrivePublisher.reset();
            m_DriveControlSubscription.reset();
            m_AEBProviders.clear();
        }

    public:
        void OnInit();

        virtual void RegisterInterruptListener() override;

    private:
        struct SequentialSpeedChangeInfo {
            double MinSpeed = 0.0;
            double MaxSpeed = std::numeric_limits<double>::infinity();
            // CurrentSpeedMultiplier / MessageSpeedMultiplier define how much we bias
            // towards the current speed vs the incoming command.
            double CurrentSpeedMultiplier = 0.95;
            double MessageSpeedMultiplier = 1.0;
            // ComposeAlpha blends the incoming commanded speed and the existing/current speed.
            double ComposeAlpha = 0.0;
            double OneMinusAlpha = 1.0;
            // Mode = addition
        };

        ackermann_msgs::msg::AckermannDrive SequentialSpeedChange(const SequentialSpeedChangeInfo &info,
                                                                  ackermann_msgs::msg::AckermannDrive &&drive);

        void AEBTimerCallback();

        void ConsoleLoop();

        void StopThreads();

        SafetyState EvaluateSafetyState(const SafetyInfo &info) const;

        void LogStageChange(SafetyState new_stage, const SafetyInfo &info);

        void OnDriveCommand(dev_b7_interfaces::msg::DriveControlMessage::SharedPtr msg);

        void OnInterrupt() {
            ackermann_msgs::msg::AckermannDriveStamped stop_msg;
            stop_msg.header.stamp = this->now();
            stop_msg.header.frame_id = "base_link";
            stop_msg.drive.speed = 0.0f;
            stop_msg.drive.acceleration = 0.0f;

            m_AckermannDrivePublisher->publish(stop_msg);

            RCLCPP_INFO(get_logger(), "Interrupt signal received, published stop command.");

            m_ShouldStop.store(true, std::memory_order_release);

            GetGlobalExecutor().cancel();
        }

    private:
        struct AEBParameters {
            std::unique_ptr<DynamicParameter<double>> TTCWarning{};
            std::unique_ptr<DynamicParameter<double>> TTCPartial{};
            std::unique_ptr<DynamicParameter<double>> TTCFull{};

            std::unique_ptr<DynamicParameter<double>> DistanceWarning{};
            std::unique_ptr<DynamicParameter<double>> DistancePartial{};
            std::unique_ptr<DynamicParameter<double>> DistanceFull{};

            // std::unique_ptr<DynamicParameter<double>> PartialBrakeDecel{};
            // std::unique_ptr<DynamicParameter<double>> PartialSpeedLimit{};

            std::unique_ptr<DynamicParameter<bool>> AutoRelease{};
        } m_AEBParameters;

    private:
        std::atomic<SafetyState> m_AEBStage{SafetyState::eOptimal};
        std::atomic<bool> m_ShouldStop{false};

        std::vector<std::shared_ptr<ISafetyInfoProvider>> m_AEBProviders;

    private:
        std::mutex m_MapAccessMutex;

        std::map<int32_t, dev_b7_interfaces::msg::DriveControlMessage::SharedPtr> mPriorityToLastMessageMap;

        std::timed_mutex m_LastDriveControlMessageMutex;
        ackermann_msgs::msg::AckermannDriveStamped m_LastReceivedMessage;

    private:
        rclcpp::TimerBase::SharedPtr m_AEBTimer;
        std::thread m_ConsoleInputListenerThread;

    private:
        rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
        m_AckermannDrivePublisher;

        rclcpp::Subscription<dev_b7_interfaces::msg::DriveControlMessage>::SharedPtr m_DriveControlSubscription;

    private:
        std::array<MessageModifier, 4> m_MessageModifiers;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdometrySubscription;
        std::atomic<double> m_CurrentSpeed{0.0};
    };

    void DriveControl::OnInit() {
        INodeExtend::OnInit();

        ParameterConversion::EnableParameterMismatchComplaints(false);

        bool enableConsoleInputListener = NodeUtils::GetParameter<bool>(this, "enable_console_listener").
                value_or(false);

        m_ShouldStop.store(false, std::memory_order_release);

        auto lidar_provider = CreateLidarAEBProvider();
        lidar_provider->OnInit(this);
        m_AEBProviders.push_back(lidar_provider);

        // auto depth_map_provider = std::make_shared<AEB::DepthMapSafetyInfoProvider>(ComputeMode::eAuto);
        // depth_map_provider->OnInit(this);
        // m_AEBProviders.push_back(depth_map_provider);

        m_AEBTimer = this->create_wall_timer(
            std::chrono::milliseconds(25),
            std::bind(&DriveControl::AEBTimerCallback, this));

        if (enableConsoleInputListener)
            m_ConsoleInputListenerThread = std::thread(&DriveControl::ConsoleLoop, this);

        m_DriveControlSubscription = this->create_subscription<dev_b7_interfaces::msg::DriveControlMessage>(
            dev_b7_interfaces::msg::DriveControlMessage::BUILTIN_TOPIC_NAME_STRING, 10,
            std::bind(&DriveControl::OnDriveCommand, this, std::placeholders::_1));


        m_AckermannDrivePublisher = this->create_publisher<
            ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);

        m_OdometrySubscription = this->create_subscription<nav_msgs::msg::Odometry>(
            NodeUtils::GetParameter<bool>(this, "sim").value_or(false) ? "/ego_racecar/odom" : "/odom",
            rclcpp::SensorDataQoS(),
            [this](nav_msgs::msg::Odometry::SharedPtr msg) {
                m_CurrentSpeed.store(std::abs(msg->twist.twist.linear.x), std::memory_order_release);
            }
        );

        m_MessageModifiers.at(static_cast<size_t>(SafetyState::eOptimal)) = Identity;
        m_MessageModifiers.at(static_cast<size_t>(SafetyState::eSuboptimal)) = Identity;
        m_MessageModifiers.at(static_cast<size_t>(SafetyState::eCritical)) = MessageModifier(
            [node = weak_from_this()](ackermann_msgs::msg::AckermannDrive &&msg) {
                //std::cout << "called\n";
                if (auto locked = node.lock()) {
                    //std::cout << "locked\n";
                    return std::static_pointer_cast<DriveControl>(locked)->SequentialSpeedChange(
                        SequentialSpeedChangeInfo{
                            .MinSpeed = 0.0,
                            .MaxSpeed = std::numeric_limits<double>::infinity(),
                            .CurrentSpeedMultiplier = 0.98,
                            .MessageSpeedMultiplier = 1.0,
                            .ComposeAlpha = 0.4,
                            .OneMinusAlpha = 0.6
                        }, std::move(msg));
                }
                return msg;
            });
        m_MessageModifiers.at(static_cast<size_t>(SafetyState::eEmergency)) = MessageModifier(
            [](ackermann_msgs::msg::AckermannDrive &&msg) {
                msg.speed = 0.0;
                msg.acceleration = 10.0;
                return msg;
            });
    }


    ackermann_msgs::msg::AckermannDrive DriveControl::SequentialSpeedChange(const SequentialSpeedChangeInfo &info,
                                                                            ackermann_msgs::msg::AckermannDrive &&
                                                                            drive) {
        // Update the command with a simple blended speed target, then clamp it.
        // The intent is to avoid abrupt speed jumps when safety logic changes stage.
        drive.acceleration = 1.0;
        drive.speed = std::min<float>(static_cast<float>(std::clamp<double>(
            m_CurrentSpeed * info.CurrentSpeedMultiplier * info.OneMinusAlpha
            + drive.speed * info.MessageSpeedMultiplier * info.ComposeAlpha,
            info.MinSpeed, info.MaxSpeed)), drive.speed);
            
        // std::cout << "SequentialSpeedChange speed: " << drive.speed << std::endl;
        
        // RCLCPP_WARN(get_logger(), "SequentialSpeedChange speed: %f.", drive.speed);
         
        return drive;
    }

    void DriveControl::AEBTimerCallback() {
        SafetyInfo fused_info;
        fused_info.Source = "fusion";
        // Start with infinities, then take the minimum across all providers.
        fused_info.MinDistance = std::numeric_limits<double>::infinity();
        fused_info.MinTTC = std::numeric_limits<double>::infinity();
        fused_info.Timestamp = this->now();

        for (const auto &provider: m_AEBProviders) {
            SafetyInfo info = provider->GetSafetyInfo();

            if (info.MinDistance < fused_info.MinDistance) {
                fused_info.MinDistance = info.MinDistance;
            }

            if (info.MinTTC < fused_info.MinTTC) {
                fused_info.MinTTC = info.MinTTC;
            }
        }

        SafetyState new_stage = EvaluateSafetyState(fused_info);

        SafetyState old_stage = m_AEBStage.load(std::memory_order_acquire);
        if (!m_AEBParameters.AutoRelease->GetCopied() && old_stage == SafetyState::eEmergency) {
            // If auto-release is disabled, once emergency is reached we stay in emergency.
            new_stage = SafetyState::eEmergency;
        }

        if (new_stage != old_stage) {
            m_AEBStage.store(new_stage, std::memory_order_release);
            LogStageChange(new_stage, fused_info);
        }

        if (new_stage == SafetyState::eOptimal || new_stage == SafetyState::eSuboptimal) {
            // For non-critical stages we forward the original drive command without modification.
            return;
        }

        ackermann_msgs::msg::AckermannDriveStamped msg;

        {
            std::unique_lock lock(m_LastDriveControlMessageMutex, std::chrono::milliseconds(10));
            if (lock.owns_lock()) {
                msg = m_LastReceivedMessage;
            }
        }

        msg.header.stamp = this->now();
        msg.header.frame_id = "base_link";

        // switch (new_stage) {
        //     case SafetyState::eEmergency:
        //         msg.drive.speed = 0.0f;
        //         msg.drive.acceleration = 0.0f;
        //         break;
        //
        //     case SafetyState::eCritical:
        //         msg.drive.acceleration = static_cast<float>(m_AEBParameters.PartialBrakeDecel->GetCopied());
        //         msg.drive.speed = static_cast<float>(
        //             std::min<float>(m_AEBParameters.PartialSpeedLimit->GetCopied(), msg.drive.speed));
        //         break;
        //
        //     case SafetyState::eOptimal:
        //     case SafetyState::eSuboptimal:
        //
        //     default:
        //
        //         break;
        // }

        msg.drive = m_MessageModifiers.at(static_cast<size_t>(new_stage))(std::move(msg.drive));

        m_AckermannDrivePublisher->publish(msg);
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
                        GetGlobalExecutor().cancel();
                    } else if (line == "aeb_on") {
                        m_AEBStage.store(SafetyState::eEmergency, std::memory_order_release);
                        RCLCPP_WARN(get_logger(), "AEB Engaged manually (FULL BRAKE).");
                    } else if (line == "aeb_off") {
                        m_AEBStage.store(SafetyState::eOptimal, std::memory_order_release);
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

        if (m_ConsoleInputListenerThread.joinable()) {
            m_ConsoleInputListenerThread.join();
        }
    }

    SafetyState DriveControl::EvaluateSafetyState(const SafetyInfo &info) const {

        SafetyState old_state = m_AEBStage.load(std::memory_order_acquire);
        double emergency_dist = m_AEBParameters.DistanceFull->GetCopied();

        if (old_state == SafetyState::eEmergency) {
            // Placeholder for hysteresis behavior: multiplier is currently 1.0,
            // so thresholds do not change while staying in eEmergency.
            emergency_dist *= 1.0;
        }

        // Threshold policy:
        // - Emergency: very low TTC OR too close to allow motion
        // - Critical: medium TTC OR medium distance
        // - Suboptimal/Warning: early warning thresholds
        // - Optimal: no threshold exceeded
        if (info.MinTTC < m_AEBParameters.TTCFull->GetCopied() || info.MinDistance < emergency_dist) {
            return SafetyState::eEmergency;
        }

        if (info.MinTTC < m_AEBParameters.TTCPartial->GetCopied() || info.MinDistance < m_AEBParameters.DistancePartial
            ->GetCopied()) {
            return SafetyState::eCritical;
        }

        if (info.MinTTC < m_AEBParameters.TTCWarning->GetCopied() || info.MinDistance < m_AEBParameters.DistanceWarning
            ->GetCopied()) {
            return SafetyState::eSuboptimal;
        }

        return SafetyState::eOptimal;
    }

    void DriveControl::LogStageChange(SafetyState new_stage, const SafetyInfo &info) {
        switch (new_stage) {
            case SafetyState::eEmergency:
                RCLCPP_ERROR(get_logger(), "[%s] AEB FULL BRAKE - TTC: %.3f s, Dist: %.3f m",
                             info.Source.c_str(), info.MinTTC, info.MinDistance);
                break;

            case SafetyState::eCritical:
                RCLCPP_WARN(get_logger(), "[%s] AEB PARTIAL BRAKE - TTC: %.3f s, Dist: %.3f m",
                            info.Source.c_str(), info.MinTTC, info.MinDistance);
                break;

            case SafetyState::eSuboptimal:
                RCLCPP_INFO(get_logger(), "[%s] AEB WARNING - TTC: %.3f s, Dist: %.3f m",
                            info.Source.c_str(), info.MinTTC, info.MinDistance);
                break;

            case SafetyState::eOptimal:
                if (m_AEBParameters.AutoRelease->GetCopied()) {
                    RCLCPP_INFO(get_logger(), "AEB Released - safe conditions restored.");
                }
                break;
        }
    }

    void DriveControl::OnDriveCommand(const dev_b7_interfaces::msg::DriveControlMessage::SharedPtr msg) {
        {
            std::lock_guard<std::mutex> lock(m_MapAccessMutex);

            // Maintain a map from priority -> last active command of that priority.
            if (msg->active) {
                mPriorityToLastMessageMap[msg->priority] = msg;
            } else {
                mPriorityToLastMessageMap.erase(msg->priority);
                return;
            }

            if (auto highest_priority_msg(mPriorityToLastMessageMap.rbegin()->second); highest_priority_msg) {
                SafetyState stage = m_AEBStage.load(std::memory_order_acquire);

                // if (stage == SafetyState::eOptimal || stage == SafetyState::eSuboptimal) {
                //     m_AckermannDrivePublisher->publish(highest_priority_msg->drive);
                // }

                // Apply the stage-specific modifier (e.g., emergency full brake, or critical speed limiting).
                auto copied = highest_priority_msg->drive;
                copied.drive = m_MessageModifiers.at(static_cast<size_t>(stage))(std::move(copied.drive));

                m_AckermannDrivePublisher->publish(std::move(copied));
            }
        }

        {
            std::lock_guard<std::timed_mutex> lock(m_LastDriveControlMessageMutex);
            m_LastReceivedMessage = msg->drive;
        }
    }

    // Implement RegisterInterruptListener
    void DriveControl::RegisterInterruptListener() {
        // The ROS node installs a SIGINT handler to trigger an emergency stop.
        // This implementation assumes only one DriveControl instance is alive
        // (static storage holds a single `instance` pointer and handler function).
        static DriveControl *instance = this; // Capture 'this' in a static variable for use in the signal handler
        static void(*original_handler)(int) = std::signal(SIGINT, [](int signum) {
            instance->OnInterrupt(); // Call the member function to handle the interrupt signal
            if (original_handler) {
                original_handler(signum);
            }
        });
    }
}

std::shared_ptr<rclcpp::Node> CreateApplicationNode() {
    auto ptr = std::make_shared<Impl::DriveControl>();
    ptr->OnInit();
    ptr->RegisterInterruptListener();
    return ptr;
}
