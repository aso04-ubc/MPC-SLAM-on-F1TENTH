#pragma once

#include "AppPCH.h"

class ILifecycleNodeEXT;

class IExecutionContext : public std::enable_shared_from_this<IExecutionContext> {
public:
    virtual ~IExecutionContext() = default;

public:
    virtual std::weak_ptr<ILifecycleNodeEXT> Attach(std::shared_ptr<ILifecycleNodeEXT> node) = 0;

    virtual void Spin() = 0;

    virtual void Cancel() = 0;

    virtual void Detach(std::weak_ptr<ILifecycleNodeEXT> node) = 0;
};

class ILifecycleNodeEXT : public rclcpp_lifecycle::LifecycleNode {
public:
    using LifecycleNode::LifecycleNode;

public:
    virtual void OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) = 0;

    virtual void OnDetachFromContext(const std::shared_ptr<IExecutionContext>& context) = 0;
};

class LifecycleNodeEXT : public ILifecycleNodeEXT {
public:
    using ILifecycleNodeEXT::ILifecycleNodeEXT;

public:
    virtual void OnAttachToContext(const std::shared_ptr<IExecutionContext>& context) override {
        m_ExecutionContext = context;
    }

    virtual void OnDetachFromContext(const std::shared_ptr<IExecutionContext>& context) override {
        m_ExecutionContext.reset();
    }

private:
    std::weak_ptr<IExecutionContext> m_ExecutionContext;
};

class ExecutionContextMultithreaded : public IExecutionContext {
public:
    ExecutionContextMultithreaded(size_t thread_count = 4);
    virtual ~ExecutionContextMultithreaded() = default;

public:
    virtual std::weak_ptr<ILifecycleNodeEXT> Attach(std::shared_ptr<ILifecycleNodeEXT> node) override;
    virtual void Detach(std::weak_ptr<ILifecycleNodeEXT> node) override;
    virtual void Spin() override;
    virtual void Cancel() override;
private:
    std::shared_ptr<rclcpp::Executor> m_Executor;
    std::vector<std::weak_ptr<ILifecycleNodeEXT>> m_AttachedNodes;
    std::mutex m_VectorAccessMutex;
};
