/**
 * @file ExecutionContext.hpp
 * @brief Execution context and lifecycle node extension interfaces
 *
 * This file defines the interfaces for managing ROS2 lifecycle nodes with
 * custom execution contexts. It provides a framework for attaching/detaching
 * nodes to/from multi-threaded executors with lifecycle management.
 */

#pragma once

#include "AppPCH.h"

// Forward declaration
class ILifecycleNodeEXT;

/**
 * @class IExecutionContext
 * @brief Interface for managing node execution contexts
 *
 * This interface defines the contract for execution contexts that can manage
 * multiple lifecycle nodes. It allows nodes to be attached/detached dynamically
 * and provides control over the execution (spin/cancel).
 */
class IExecutionContext : public std::enable_shared_from_this<IExecutionContext> {
public:
    virtual ~IExecutionContext() = default;

public:
    /**
     * @brief Attach a lifecycle node to this execution context
     *
     * Adds the node to the executor and notifies the node of the attachment.
     *
     * @param node The lifecycle node to attach
     * @return std::weak_ptr<ILifecycleNodeEXT> Weak pointer to the attached node
     */
    virtual std::weak_ptr<ILifecycleNodeEXT> Attach(std::shared_ptr<ILifecycleNodeEXT> node) = 0;

    /**
     * @brief Start spinning the executor
     *
     * Begins processing callbacks for all attached nodes. This is a blocking call
     * that runs until Cancel() is called or all nodes are detached.
     */
    virtual void Spin() = 0;

    /**
     * @brief Cancel the executor spin
     *
     * Stops the executor from processing callbacks, causing Spin() to return.
     */
    virtual void Cancel() = 0;

    /**
     * @brief Detach a lifecycle node from this execution context
     *
     * Removes the node from the executor and notifies the node of the detachment.
     * If this was the last node, the executor is automatically cancelled.
     *
     * @param node Weak pointer to the node to detach
     */
    virtual void Detach(std::weak_ptr<ILifecycleNodeEXT> node) = 0;
};

/**
 * @class ILifecycleNodeEXT
 * @brief Extended lifecycle node interface with execution context awareness
 *
 * This interface extends the standard ROS2 lifecycle node with methods to
 * handle attachment/detachment to/from execution contexts. Nodes implementing
 * this interface can be notified when they are added or removed from an executor.
 */
class ILifecycleNodeEXT : public rclcpp_lifecycle::LifecycleNode {
public:
    using LifecycleNode::LifecycleNode;

public:
    /**
     * @brief Called when the node is attached to an execution context
     *
     * Override this method to perform actions when the node is added to an executor.
     * Typically used to store a reference to the execution context.
     *
     * @param context The execution context the node was attached to
     */
    virtual void OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) = 0;

    /**
     * @brief Called when the node is detached from an execution context
     *
     * Override this method to perform cleanup when the node is removed from an executor.
     * Typically used to clear the stored reference to the execution context.
     *
     * @param context The execution context the node was detached from
     */
    virtual void OnDetachFromContext(const std::shared_ptr<IExecutionContext>& context) = 0;
};

/**
 * @class LifecycleNodeEXT
 * @brief Base implementation of ILifecycleNodeEXT with default context management
 *
 * This class provides a default implementation of the execution context
 * attachment/detachment behavior, storing a weak pointer to the context.
 * Derive from this class for simple use cases.
 */
class LifecycleNodeEXT : public ILifecycleNodeEXT {
public:
    using ILifecycleNodeEXT::ILifecycleNodeEXT;

public:
    /**
     * @brief Store a weak reference to the execution context
     * @param context The execution context being attached
     */
    virtual void OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) override {
        m_ExecutionContext = context;
    }

    /**
     * @brief Clear the stored execution context reference
     * @param context The execution context being detached
     */
    virtual void OnDetachFromContext(const std::shared_ptr<IExecutionContext>&) override {
        m_ExecutionContext.reset();
    }

private:
    std::weak_ptr<IExecutionContext> m_ExecutionContext; ///< Weak reference to avoid circular dependency
};

/**
 * @class ExecutionContextMultithreaded
 * @brief Multi-threaded execution context implementation
 *
 * This class implements IExecutionContext using a ROS2 MultiThreadedExecutor.
 * It manages multiple lifecycle nodes and processes their callbacks across
 * multiple threads for improved performance.
 */
class ExecutionContextMultithreaded : public IExecutionContext {
public:
    /**
     * @brief Construct a multi-threaded execution context
     * @param thread_count Number of threads for the executor (default: 4)
     */
    ExecutionContextMultithreaded(size_t thread_count = 4);

    virtual ~ExecutionContextMultithreaded() = default;

public:
    virtual std::weak_ptr<ILifecycleNodeEXT> Attach(std::shared_ptr<ILifecycleNodeEXT> node) override;
    virtual void Detach(std::weak_ptr<ILifecycleNodeEXT> node) override;
    virtual void Spin() override;
    virtual void Cancel() override;

private:
    std::shared_ptr<rclcpp::Executor> m_Executor;              ///< The multi-threaded executor instance
    std::vector<std::weak_ptr<ILifecycleNodeEXT>> m_AttachedNodes; ///< List of attached nodes
    std::mutex m_VectorAccessMutex;                            ///< Mutex for thread-safe node list access
};
