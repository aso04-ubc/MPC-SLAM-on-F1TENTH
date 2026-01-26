#include "ExecutionContext.hpp"

ExecutionContextMultithreaded::ExecutionContextMultithreaded(size_t thread_count) {
    rclcpp::ExecutorOptions options;
    m_Executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(options, thread_count);
}

std::weak_ptr<ILifecycleNodeEXT> ExecutionContextMultithreaded::Attach(std::shared_ptr<ILifecycleNodeEXT> node) {
    m_Executor->add_node(node->get_node_base_interface());
    node->OnAttachToContext(shared_from_this());
    std::lock_guard lock(m_VectorAccessMutex);
    m_AttachedNodes.push_back(node);
    return node;
}

void ExecutionContextMultithreaded::Detach(std::weak_ptr<ILifecycleNodeEXT> node) {
    if (auto shared_node = node.lock()) {
        m_Executor->remove_node(shared_node->get_node_base_interface());
        shared_node->OnDetachFromContext(shared_from_this());
        std::lock_guard lock(m_VectorAccessMutex);
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

void ExecutionContextMultithreaded::Spin() {
    m_Executor->spin();
}

void ExecutionContextMultithreaded::Cancel() {
    m_Executor->cancel();
}
