/**
 * @file ExecutionContext.cpp
 * @brief Implementation of multi-threaded execution context
 *
 * This file provides the implementation of ExecutionContextMultithreaded,
 * which manages lifecycle nodes in a multi-threaded ROS2 executor environment.
 */

#include "ExecutionContext.hpp"

/**
 * @brief Construct a multi-threaded execution context
 *
 * Creates a ROS2 MultiThreadedExecutor with the specified number of threads.
 * Each thread can independently process callbacks from attached nodes,
 * improving throughput for multiple concurrent operations.
 *
 * @param thread_count Number of worker threads (default: 4)
 */
ExecutionContextMultithreaded::ExecutionContextMultithreaded(size_t thread_count) {
    rclcpp::ExecutorOptions options;
    m_Executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(options, thread_count);
}

/**
 * @brief Attach a lifecycle node to the executor
 *
 * STEPS:
 * 1. Add the node to the underlying ROS2 executor
 * 2. Notify the node it has been attached (allows node to store context reference)
 * 3. Add node to our internal tracking list (thread-safe with mutex)
 * 4. Return a weak pointer to prevent circular references
 *
 * @param node Shared pointer to the node to attach
 * @return std::weak_ptr<ILifecycleNodeEXT> Weak pointer to the attached node
 */
std::weak_ptr<ILifecycleNodeEXT> ExecutionContextMultithreaded::Attach(std::shared_ptr<ILifecycleNodeEXT> node) {
    m_Executor->add_node(node->get_node_base_interface());
    node->OnAttachToContext(shared_from_this());
    std::lock_guard<std::mutex> lock(m_VectorAccessMutex);
    m_AttachedNodes.push_back(node);
    return node;
}

/**
 * @brief Detach a lifecycle node from the executor
 *
 * STEPS:
 * 1. Lock the weak pointer to get a shared pointer (fails if node already destroyed)
 * 2. Remove node from the ROS2 executor
 * 3. Notify the node it has been detached (allows node to clear context reference)
 * 4. Remove node from our internal tracking list (thread-safe with mutex)
 * 5. If this was the last node, automatically cancel the executor
 *
 * The automatic cancellation ensures Spin() returns when there are no more nodes
 * to process, allowing for graceful shutdown.
 *
 * @param node Weak pointer to the node to detach
 */
void ExecutionContextMultithreaded::Detach(std::weak_ptr<ILifecycleNodeEXT> node) {
    if (auto shared_node = node.lock()) {
        m_Executor->remove_node(shared_node->get_node_base_interface());
        shared_node->OnDetachFromContext(shared_from_this());
        std::lock_guard<std::mutex> lock(m_VectorAccessMutex);
        m_AttachedNodes.erase(
            std::remove_if(
                m_AttachedNodes.begin(),
                m_AttachedNodes.end(),
                [&shared_node](const std::weak_ptr<ILifecycleNodeEXT>& n) {
                    return n.lock() == shared_node;
                }),
            m_AttachedNodes.end());
        if (m_AttachedNodes.empty()) {
            Cancel();
        }
    }
}

/**
 * @brief Start spinning the executor (blocking)
 *
 * This is a blocking call that continuously processes callbacks for all
 * attached nodes across multiple threads. Returns when Cancel() is called
 * or when an interrupt occurs.
 */
void ExecutionContextMultithreaded::Spin() {
    m_Executor->spin();
}

/**
 * @brief Cancel the executor spin
 *
 * Signals the executor to stop processing callbacks, causing Spin() to return.
 * This is typically called on shutdown or when all nodes have been detached.
 */
void ExecutionContextMultithreaded::Cancel() {
    m_Executor->cancel();
}
