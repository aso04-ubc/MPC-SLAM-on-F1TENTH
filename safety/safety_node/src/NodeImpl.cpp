/**
 * @file NodeImpl.cpp
 * @brief Implementation of the Drive Control Node with AEB (Autonomous Emergency Braking)
 *
 * OVERVIEW:
 * =========
 * This node manages drive commands with priority-based arbitration and provides
 * autonomous emergency braking functionality. It subscribes to multiple topics,
 * arbitrates between competing drive commands, and can override commands with
 * emergency braking when collision is imminent.
 *
 * KEY FEATURES:
 * =============
 * 1. PRIORITY-BASED COMMAND ARBITRATION
 *    - Multiple sources can send drive commands with different priorities
 *    - Higher priority commands override lower priority ones
 *    - Commands can be activated/deactivated dynamically
 *
 * 2. AUTONOMOUS EMERGENCY BRAKING (AEB)
 *    - Monitors laser scan data for obstacles
 *    - Calculates Time-To-Collision (TTC) for each scan point
 *    - Engages emergency braking when TTC < threshold OR distance < threshold
 *    - Overrides all drive commands when active
 *
 * 3. LIFECYCLE MANAGEMENT
 *    - Implements ROS2 lifecycle states (configure, activate, deactivate, cleanup)
 *    - Proper thread management during transitions
 *
 * 4. CONSOLE CONTROL
 *    - Interactive commands via stdin: "exit", "aeb_on", "aeb_off"
 *    - Non-blocking polling of console input
 *
 * SUBSCRIBED TOPICS:
 * ==================
 * - /drive_control (dev_b7_interfaces::msg::DriveControlMessage)
 *   Drive commands with priority levels and active/inactive state
 *
 * - /scan (sensor_msgs::msg::LaserScan)
 *   Lidar data for obstacle detection and TTC calculation
 *
 * - /ego_racecar/odom (nav_msgs::msg::Odometry)
 *   Vehicle odometry for current speed (used in TTC calculation)
 *
 * PUBLISHED TOPICS:
 * =================
 * - /drive (ackermann_msgs::msg::AckermannDriveStamped)
 *   Final drive commands sent to the vehicle controller
 *
 * THREADS:
 * ========
 * 1. Main ROS2 callback threads (executor managed)
 * 2. AEB Loop Thread: Publishes emergency stop commands when AEB is active
 * 3. Console Loop Thread: Monitors stdin for user commands
 */

#include "Application.hpp"

namespace Impl {
    using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    /**
     * @class DriveControl
     * @brief Main drive control node implementation with AEB functionality
     *
     * This class implements a lifecycle node that manages drive commands and
     * provides autonomous emergency braking. It uses priority-based arbitration
     * to select between multiple drive command sources and can override any
     * command with emergency braking.
     */
    class DriveControl : public ILifecycleNodeEXT {
    public:
        /**
         * @brief Constructor for DriveControl node
         *
         * Initializes the node with AEB thresholds from configuration.
         * Does not start threads or create publishers/subscribers yet
         * (those happen in on_configure and on_activate).
         *
         * @param info Configuration containing AEB thresholds
         */
        DriveControl(const NodeCreationInfo& info)
            : ILifecycleNodeEXT("drive_control_node"),
              TTCThreshold(info.aeb_ttc_threshold),
              DistanceThreshold(info.aeb_minimum_distance) {
            RCLCPP_INFO(get_logger(), "DriveControl Node Object Created.");
        }

        /**
         * @brief Destructor - ensures threads are stopped
         */
        virtual ~DriveControl() {
            StopThreads();
        }

        // These are called by the ROS2 lifecycle management system
        // to transition the node between states

        /**
         * @brief Configure state callback - Initialize resources
         *
         * Creates publishers and subscribers. Does not start threads yet.
         *
         * @return SUCCESS if configuration succeeded
         */
        CallbackReturn on_configure(const rclcpp_lifecycle::State&) override;

        /**
         * @brief Activate state callback - Start processing
         *
         * Activates the publisher, starts AEB and console threads.
         * The node begins processing messages and performing AEB checks.
         *
         * @return SUCCESS if activation succeeded
         */
        CallbackReturn on_activate(const rclcpp_lifecycle::State&) override;

        /**
         * @brief Deactivate state callback - Stop processing
         *
         * Deactivates the publisher and stops all threads gracefully.
         * Subscriptions remain but callbacks won't trigger publications.
         *
         * @return SUCCESS if deactivation succeeded
         */
        CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override;

        /**
         * @brief Cleanup state callback - Release resources
         *
         * Destroys all publishers and subscriptions, freeing resources.
         *
         * @return SUCCESS if cleanup succeeded
         */
        CallbackReturn on_cleanup(const rclcpp_lifecycle::State&) override;

        /**
         * @brief Shutdown state callback - Final cleanup
         *
         * Called when the node is shutting down. Ensures threads are stopped.
         *
         * @return SUCCESS if shutdown succeeded
         */
        CallbackReturn on_shutdown(const rclcpp_lifecycle::State&) override;

    private:

        /**
         * @brief AEB loop that runs in a separate thread
         *
         * When AEB is active, this thread continuously publishes stop commands
         * at 200 Hz (every 5ms) to ensure the vehicle stops quickly.
         * Also preserves the steering angle from the last command.
         */
        void AEBLoop();

        /**
         * @brief Console input loop that runs in a separate thread
         *
         * Non-blocking polling of stdin for user commands:
         * - "exit" or "quit": Shutdown the node
         * - "aeb_on": Manually engage AEB
         * - "aeb_off": Manually disengage AEB
         *
         * Uses poll() with 100ms timeout to avoid blocking forever.
         */
        void ConsoleLoop();

        /**
         * @brief Stop all worker threads gracefully
         *
         * Sets the stop flag and joins both threads.
         * Safe to call multiple times.
         */
        void StopThreads();


        /**
         * @brief Callback for odometry messages
         *
         * Updates the current vehicle speed, which is used in TTC calculations.
         * Uses atomic store for thread-safe access from the laser scan callback.
         *
         * @param msg Odometry message with vehicle velocity
         */
        void OnProcessOdometry(const nav_msgs::msg::Odometry::SharedPtr msg) {
            m_CurrentSpeed.store(msg->twist.twist.linear.x, std::memory_order_release);
        }

        /**
         * @brief Callback for laser scan messages
         *
         * Processes lidar data to detect potential collisions:
         * 1. For each valid scan point, calculate distance and angle
         * 2. Compute range rate (closing speed) = vehicle_speed * cos(angle)
         * 3. Calculate Time-To-Collision (TTC) = distance / range_rate
         * 4. Engage AEB if TTC < threshold OR distance < threshold
         * 5. Disengage AEB if all obstacles are beyond thresholds
         *
         * @param msg LaserScan message with range measurements
         */
        void OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg);

        /**
         * @brief Callback for drive control command messages
         *
         * Handles priority-based arbitration of drive commands:
         * 1. If command is active, add/update it in the priority map
         * 2. If command is inactive, remove it from the priority map
         * 3. Select the highest priority active command
         * 4. Publish the command (unless AEB is active)
         * 5. Store the command for AEB to preserve steering angle
         *
         * @param msg Drive control message with priority and active flag
         */
        void OnDriveCommand(const dev_b7_interfaces::msg::DriveControlMessage::SharedPtr msg);

    private:

        const double TTCThreshold; ///< Time-To-Collision threshold for AEB (seconds)
        const double DistanceThreshold; ///< Minimum distance threshold for AEB (meters)

    private:
        // These are accessed from multiple threads, so we use atomics

        std::atomic<double> m_CurrentSpeed{0.0}; ///< Current vehicle speed from odometry
        std::atomic<bool> m_IsAEBActive{false}; ///< Whether AEB is currently engaged
        std::atomic<bool> m_ShouldStop{false}; ///< Flag to signal threads to stop

    private:

        std::mutex m_MapAccessMutex; ///< Protects the priority map

        /**
         * @brief Map from priority level to last received command at that priority
         *
         * Higher priority values take precedence. The map is automatically sorted
         * by key, so rbegin() gives the highest priority command.
         */
        std::map<int32_t, dev_b7_interfaces::msg::DriveControlMessage::SharedPtr> mPriorityToLastMessageMap;

        std::timed_mutex m_LastDriveControlMessageMutex; ///< Protects last received message
        ackermann_msgs::msg::AckermannDriveStamped m_LastReceivedMessage;
        ///< Last command (for AEB steering preservation)

    private:

        std::thread m_AEBSubmissionThread; ///< Thread that publishes stop commands when AEB is active
        std::thread m_ConsoleInputListenerThread; ///< Thread that listens for console commands

    private:

        /**
         * @brief Publisher for Ackermann drive commands
         * Lifecycle publisher that can be activated/deactivated
         */
        rclcpp_lifecycle::LifecyclePublisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr
        m_AckermannDrivePublisher;

        rclcpp::Subscription<dev_b7_interfaces::msg::DriveControlMessage>::SharedPtr m_DriveControlSubscription;
        rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr m_ScanSubscription;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr m_OdomSubscription;

    private:

        std::weak_ptr<IExecutionContext> m_CurrentExecutionContext; ///< Weak reference to the executor

    public:

        virtual void OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) override;

        virtual void OnDetachFromContext(const std::shared_ptr<IExecutionContext>& context) override;
    };

    /**
     * @brief Configure lifecycle state - Initialize publishers and subscriptions
     *
     * This callback is invoked when the node transitions to the CONFIGURED state.
     * It creates all ROS2 communication objects but does not start threads or
     * activate the publisher.
     *
     * CREATED OBJECTS:
     * ================
     * - Publisher: /drive (AckermannDriveStamped) - Final drive commands
     * - Subscription: /drive_control (DriveControlMessage) - Input commands with priority
     * - Subscription: /scan (LaserScan) - Lidar data for AEB
     * - Subscription: /ego_racecar/odom (Odometry) - Vehicle speed
     *
     * @param state The previous lifecycle state
     * @return CallbackReturn::SUCCESS if configuration succeeded
     */
    CallbackReturn DriveControl::on_configure(const rclcpp_lifecycle::State&) {
        RCLCPP_INFO(get_logger(), "Configuring: Initializing Subscriptions and Publishers...");

        // Create lifecycle publisher for drive commands
        m_AckermannDrivePublisher = this->create_publisher<
            ackermann_msgs::msg::AckermannDriveStamped>("/drive", 10);

        // Subscribe to drive control commands (with priority arbitration)
        m_DriveControlSubscription = this->create_subscription<dev_b7_interfaces::msg::DriveControlMessage>(
            dev_b7_interfaces::msg::DriveControlMessage::BUILTIN_TOPIC_NAME_STRING, 10,
            std::bind(&DriveControl::OnDriveCommand, this, std::placeholders::_1));

        // Subscribe to laser scan for AEB collision detection
        m_ScanSubscription = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&DriveControl::OnProcessLaserScan, this, std::placeholders::_1));

        // Subscribe to odometry for vehicle speed
        m_OdomSubscription = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10, std::bind(&DriveControl::OnProcessOdometry, this, std::placeholders::_1));

        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief Activate lifecycle state - Start processing
     *
     * This callback is invoked when the node transitions to the ACTIVE state.
     * It starts the worker threads and activates the publisher so messages
     * can be sent.
     *
     * ACTIONS:
     * ========
     * 1. Clear the stop flag for threads
     * 2. Start AEB loop thread (publishes stop commands when AEB active)
     * 3. Start console input thread (monitors stdin for user commands)
     * 4. Activate the lifecycle publisher (enable message publication)
     *
     * @param state The previous lifecycle state
     * @return CallbackReturn::SUCCESS if activation succeeded
     */
    CallbackReturn DriveControl::on_activate(const rclcpp_lifecycle::State&) {
        RCLCPP_INFO(get_logger(), "Activating: Starting Control Threads...");

        // Reset the stop flag
        m_ShouldStop.store(false, std::memory_order_release);

        // Start worker threads
        m_AEBSubmissionThread = std::thread(&DriveControl::AEBLoop, this);
        m_ConsoleInputListenerThread = std::thread(&DriveControl::ConsoleLoop, this);

        // Activate the publisher (required for lifecycle publishers)
        m_AckermannDrivePublisher->on_activate();

        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief Deactivate lifecycle state - Stop processing
     *
     * This callback is invoked when the node transitions to the INACTIVE state.
     * It deactivates the publisher and stops all worker threads gracefully.
     *
     * ACTIONS:
     * ========
     * 1. Deactivate the lifecycle publisher (prevent message publication)
     * 2. Stop all worker threads (sets stop flag, waits for threads to exit)
     *
     * Subscriptions remain active but the publisher won't send messages.
     *
     * @param state The previous lifecycle state
     * @return CallbackReturn::SUCCESS if deactivation succeeded
     */
    CallbackReturn DriveControl::on_deactivate(const rclcpp_lifecycle::State&) {
        RCLCPP_INFO(get_logger(), "Deactivating: Stopping Threads...");

        // Deactivate the publisher
        m_AckermannDrivePublisher->on_deactivate();

        // Stop worker threads gracefully
        StopThreads();

        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief Cleanup lifecycle state - Release resources
     *
     * This callback is invoked when the node transitions to the UNCONFIGURED state.
     * It destroys all publishers and subscriptions, releasing their resources.
     *
     * @param state The previous lifecycle state
     * @return CallbackReturn::SUCCESS if cleanup succeeded
     */
    CallbackReturn DriveControl::on_cleanup(const rclcpp_lifecycle::State&) {
        RCLCPP_INFO(get_logger(), "Cleaning up resources...");

        // Reset all communication objects (releases resources)
        m_AckermannDrivePublisher.reset();
        m_DriveControlSubscription.reset();
        m_ScanSubscription.reset();
        m_OdomSubscription.reset();

        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief Shutdown lifecycle state - Final cleanup
     *
     * This callback is invoked when the node is shutting down completely.
     * It ensures all threads are stopped before the node is destroyed.
     *
     * @param state The previous lifecycle state
     * @return CallbackReturn::SUCCESS if shutdown succeeded
     */
    CallbackReturn DriveControl::on_shutdown(const rclcpp_lifecycle::State&) {
        RCLCPP_INFO(get_logger(), "Shutting down node...");
        StopThreads();
        return CallbackReturn::SUCCESS;
    }

    /**
     * @brief AEB loop implementation
     *
     * OPERATION:
     * ==========
     * This function runs in a dedicated thread and continuously checks if AEB
     * is active. When active, it publishes stop commands at high frequency
     * (200 Hz) to ensure rapid vehicle deceleration.
     *
     * THREAD SAFETY:
     * ==============
     * - Uses atomic loads for m_IsAEBActive and m_ShouldStop
     * - Uses timed mutex for accessing last received message
     * - If mutex can't be acquired in 10ms, publishes empty stop command
     *
     * STOP COMMAND BEHAVIOR:
     * ======================
     * - Sets speed to 0.0 (full stop)
     * - Preserves steering angle from last drive command (smooth steering)
     * - Updates timestamp and frame_id for each message
     * - Clears the stored message after using it
     */
    void DriveControl::AEBLoop() {
        // Run until shutdown or stop signal
        while (rclcpp::ok() && !m_ShouldStop.load(std::memory_order_acquire)) {
            // Check if AEB is currently engaged
            if (m_IsAEBActive.load(std::memory_order_acquire)) {
                ackermann_msgs::msg::AckermannDriveStamped stop_msg;

                // Try to get the last drive command to preserve steering angle
                {
                    std::unique_lock lock(m_LastDriveControlMessageMutex, std::chrono::milliseconds(10));
                    if (lock.owns_lock()) {
                        stop_msg = m_LastReceivedMessage;
                        // m_LastReceivedMessage = ackermann_msgs::msg::AckermannDriveStamped{};
                    }
                }

                // Set message metadata
                stop_msg.header.stamp = this->now();
                stop_msg.header.frame_id = "base_link";

                // Override speed to 0 (emergency stop)
                stop_msg.drive.speed = 0.0;
                stop_msg.drive.acceleration = 0.0;

                // Publish the stop command
                m_AckermannDrivePublisher->publish(stop_msg);
            }

            // Sleep for 5ms (200 Hz update rate)
            // High frequency ensures quick response
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    /**
     * @brief Console input loop implementation
     *
     * OPERATION:
     * ==========
     * This function runs in a dedicated thread and monitors stdin for user
     * commands. It uses non-blocking polling to avoid hanging forever if
     * no input is available.
     *
     * SUPPORTED COMMANDS:
     * ===================
     * - "exit" or "quit": Detach from executor and shutdown node
     * - "aeb_on": Manually engage AEB (for testing)
     * - "aeb_off": Manually disengage AEB (for testing)
     * - Any other input: Print "Unknown command" message
     *
     * POLLING MECHANISM:
     * ==================
     * Uses poll() with 100ms timeout to check for stdin input
     * - Returns > 0 if input is available
     * - Returns 0 on timeout (no input)
     * - Returns < 0 on error (except EINTR which is retried)
     *
     * THREAD SAFETY:
     * ==============
     * - Uses atomic operations for AEB state and stop flag
     * - Safe to call from any thread
     */
    void DriveControl::ConsoleLoop() {
        // Setup pollfd structure for stdin monitoring
        pollfd fd{};
        fd.fd = STDIN_FILENO; // Monitor standard input
        fd.events = POLLIN; // Wait for input available event

        // Run until shutdown or stop signal
        while (rclcpp::ok() && !m_ShouldStop.load(std::memory_order_acquire)) {
            // Poll stdin with 100ms timeout
            int ret = poll(&fd, 1, 100);

            // Handle poll errors
            if (ret < 0) {
                if (errno == EINTR) continue; // Interrupted by signal, retry
                break; // Other error, exit loop
            }

            // Check if input is available
            if (ret > 0 && fd.revents & POLLIN) {
                std::string line;
                if (std::getline(std::cin, line)) {
                    // Command: exit or quit
                    if (line == "exit" || line == "quit") {
                        m_ShouldStop.store(true, std::memory_order_release);
                        // Detach from executor to trigger shutdown
                        if (auto executor = m_CurrentExecutionContext.lock()) {
                            executor->Detach(std::static_pointer_cast<ILifecycleNodeEXT>(shared_from_this()));
                        }
                    }
                    // Command: manually engage AEB
                    else if (line == "aeb_on") {
                        m_IsAEBActive.store(true, std::memory_order_release);
                        RCLCPP_WARN(get_logger(), "AEB Engaged manually.");
                    }
                    // Command: manually disengage AEB
                    else if (line == "aeb_off") {
                        m_IsAEBActive.store(false, std::memory_order_release);
                        RCLCPP_INFO(get_logger(), "AEB Released manually.");
                    }
                    // Unknown command
                    else if (!line.empty()) {
                        RCLCPP_INFO(get_logger(), "Unknown command: %s", line.c_str());
                    }
                }
            }
        }
        RCLCPP_INFO(get_logger(), "ConsoleLoop thread exiting.");
    }

    /**
     * @brief Stop all worker threads gracefully
     *
     * OPERATION:
     * ==========
     * 1. Set the atomic stop flag to signal threads to exit their loops
     * 2. Wait for AEB thread to finish (join)
     * 3. Wait for console thread to finish (join)
     *
     * THREAD SAFETY:
     * ==============
     * - Safe to call multiple times (joinable() check prevents double join)
     * - Blocks until both threads have exited
     * - Uses atomic flag to ensure memory visibility across threads
     */
    void DriveControl::StopThreads() {
        // Signal both threads to stop
        m_ShouldStop.store(true, std::memory_order_release);

        // Wait for AEB thread to finish if it was started
        if (m_AEBSubmissionThread.joinable()) {
            m_AEBSubmissionThread.join();
        }

        // Wait for console thread to finish if it was started
        if (m_ConsoleInputListenerThread.joinable()) {
            m_ConsoleInputListenerThread.join();
        }
    }

    /**
     * @brief Process laser scan data for AEB collision detection
     *
     * OPERATION:
     * ==========
     * This callback is invoked when new lidar data arrives. It calculates
     * Time-To-Collision (TTC) for all valid scan points to detect imminent
     * collisions.
     *
     * ALGORITHM:
     * ==========
     * For each scan point:
     * 1. Check if range is valid (not NaN/Inf, within sensor limits)
     * 2. Calculate angle = angle_min + (index * angle_increment)
     * 3. Calculate range_rate = vehicle_speed * cos(angle)
     *    - This gives the closing speed toward the obstacle
     *    - Positive means moving toward obstacle
     * 4. Calculate TTC = distance / range_rate
     *    - Only if range_rate > 0 (approaching obstacle)
     * 5. Track minimum TTC and minimum distance across all points
     *
     * AEB ENGAGEMENT:
     * ===============
     * - Engage AEB if: (min_ttc < TTCThreshold) OR (min_distance < DistanceThreshold)
     * - Disengage AEB if: Both conditions are false (all obstacles are safe)
     * - Logs ERROR when AEB engages, INFO when it releases
     *
     * @param msg LaserScan message containing range measurements
     */
    void DriveControl::OnProcessLaserScan(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // Get current vehicle speed (thread-safe atomic read)
        double v = m_CurrentSpeed.load(std::memory_order_acquire);

        // Initialize tracking variables
        double min_ttc = std::numeric_limits<double>::infinity();
        double min_distance = std::numeric_limits<double>::infinity();

        // Iterate through all laser scan points
        for (size_t i = 0; i < msg->ranges.size(); ++i) {
            double r = msg->ranges[i];

            // Skip invalid measurements
            if (std::isnan(r) || std::isinf(r) || r < msg->range_min || r > msg->range_max) {
                continue;
            }

            // Track the closest obstacle
            min_distance = std::min(min_distance, r);

            // Calculate the angle of this scan point
            double angle = msg->angle_min + i * msg->angle_increment;

            // Calculate range rate (closing speed toward obstacle)
            // cos(angle) projects vehicle velocity onto the ray direction
            double range_rate = v * std::cos(angle);

            // Only calculate TTC if we're moving toward the obstacle
            if (range_rate > 0) {
                double ttc = r / range_rate;
                if (ttc < min_ttc) {
                    min_ttc = ttc;
                }
            }
        }

        // Check if AEB should be engaged
        if (min_ttc < TTCThreshold || min_distance < DistanceThreshold) {
            // Engage AEB if not already active
            if (!m_IsAEBActive.load(std::memory_order_acquire)) {
                RCLCPP_ERROR(get_logger(), "AEB TRIGGERED! TTC: %.3f s, Dist: %.3f m", min_ttc, min_distance);
                m_IsAEBActive.store(true, std::memory_order_release);
            }
        } else {
            // Disengage AEB if currently active
            if (m_IsAEBActive.load(std::memory_order_acquire)) {
                m_IsAEBActive.store(false, std::memory_order_release);
                RCLCPP_INFO(get_logger(), "AEB Released.");
            }
        }
    }

    /**
     * @brief Process drive control command with priority-based arbitration
     *
     * OPERATION:
     * ==========
     * This callback handles incoming drive commands from multiple sources.
     * It maintains a priority map to arbitrate between competing commands
     * and publishes the highest priority active command.
     *
     * PRIORITY ARBITRATION:
     * =====================
     * 1. If command is ACTIVE:
     *    - Add/update command in priority map at its priority level
     *    - Get the highest priority command (map.rbegin())
     *    - Publish it (unless AEB is active)
     *
     * 2. If command is INACTIVE:
     *    - Remove command from priority map at its priority level
     *    - Do not publish anything (other commands may still be active)
     *
     * MESSAGE STORAGE:
     * ================
     * Always store the received message for AEB to use. This ensures AEB
     * can preserve the steering angle when publishing stop commands.
     *
     * THREAD SAFETY:
     * ==============
     * - Uses mutex for priority map access (multiple threads may call this)
     * - Uses timed_mutex for last message storage
     * - AEB state checked with atomic load
     *
     * NOTE: The goto statement is used for control flow to ensure message
     * storage happens regardless of whether the command was active or inactive.
     *
     * @param msg Drive control message with priority, active flag, and drive command
     */
    void DriveControl::OnDriveCommand(const dev_b7_interfaces::msg::DriveControlMessage::SharedPtr msg) {
        {
            // Lock the priority map for thread-safe access
            std::lock_guard<std::mutex> lock(m_MapAccessMutex);

            if (msg->active) {
                // Command is ACTIVE: add/update it in the priority map
                mPriorityToLastMessageMap[msg->priority] = msg;
            } else {
                // Command is INACTIVE: remove it from the priority map
                mPriorityToLastMessageMap.erase(msg->priority);
                // Jump to message storage (don't publish inactive commands)
                goto handle_message_store;
            }

            // Select and publish the highest priority command
            // rbegin() gives the last element (highest key/priority)
            if (auto highest_priority_msg(
                    mPriorityToLastMessageMap.rbegin()->second);
                highest_priority_msg && !m_IsAEBActive.load(std::memory_order_acquire)) {
                // Publish the highest priority command (if AEB is not active)
                m_AckermannDrivePublisher->publish(highest_priority_msg->drive);
            }
        }

    handle_message_store: // I know this is ugly, but it ensures storage always happens

        {
            // Store the message for AEB to use (preserves steering angle)
            std::lock_guard<std::timed_mutex> lock(m_LastDriveControlMessageMutex);
            m_LastReceivedMessage = msg->drive;
        }
    }

    /**
     * @brief Called when the node is attached to an execution context
     *
     * Stores a weak reference to the execution context so the node can
     * detach itself when needed (e.g., in response to "exit" console command).
     *
     * @param context The execution context this node was attached to
     */
    void DriveControl::OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) {
        m_CurrentExecutionContext = context;
    }

    /**
     * @brief Called when the node is detached from an execution context
     *
     * Clears the stored reference to the execution context.
     *
     * @param context The execution context this node was detached from
     */
    void DriveControl::OnDetachFromContext(const std::shared_ptr<IExecutionContext>&) {
        m_CurrentExecutionContext.reset();
    }
}

/**
 * @brief Factory function implementation
 *
 * Creates and returns an instance of the DriveControl node with the
 * specified configuration parameters.
 *
 * @param creation_info Configuration for AEB thresholds
 * @return std::shared_ptr<ILifecycleNodeEXT> Pointer to the created node
 */
std::shared_ptr<ILifecycleNodeEXT> CreateApplicationNode(const NodeCreationInfo& creation_info) {
    return std::make_shared<Impl::DriveControl>(creation_info);
}
