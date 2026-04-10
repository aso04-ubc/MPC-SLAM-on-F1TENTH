#pragma once

#include "AppPCH.h"

#include "Core/Interfaces.hpp"

class INodeExtend : public rclcpp::Node {
public:
    INodeExtend(const std::string &node_name, const std::string &namespace_ = "",
                const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : rclcpp::Node(node_name, namespace_, options) {
    }

    // Registers a listener for a specific parameter name.
    // Notes:
    // - The underlying callback can be invoked from ROS 2 threads.
    // - Access to listener storage is protected by m_MapAccessMutex.
    void RegisterParameterChangeListener(const std::string &parameterName, IParameterChangeListener *listener) {
        std::lock_guard<std::mutex> lock(m_MapAccessMutex);
        m_ParameterChangeListeners[parameterName].insert(listener);
    }

    // Unregisters a previously registered listener for a parameter.
    void UnregisterParameterChangeListener(const std::string &parameterName, IParameterChangeListener *listener) {
        std::lock_guard<std::mutex> lock(m_MapAccessMutex);
        if (auto it = m_ParameterChangeListeners.find(parameterName); it != m_ParameterChangeListeners.end()) {
            it->second.erase(listener);
            if (it->second.empty()) {
                m_ParameterChangeListeners.erase(it);
            }
        }
    }

    virtual void OnInit() {
        m_ParameterChangeCallbackHandle = add_on_set_parameters_callback(
            [this](const std::vector<rclcpp::Parameter> &parameters) -> rcl_interfaces::msg::SetParametersResult {
                std::lock_guard<std::mutex> lock(m_MapAccessMutex);
                for (const auto &param: parameters) {
                    if (auto it = m_ParameterChangeListeners.find(param.get_name());
                        it != m_ParameterChangeListeners.end()) {
                        for (auto *listener: m_ParameterChangeListeners[param.get_name()]) {
                            listener->OnParameterChanged(param);
                        }
                    }
                }
                rcl_interfaces::msg::SetParametersResult result{};
                result.successful = true;
                return result;
            });
    }

protected:
    OnSetParametersCallbackHandle::SharedPtr m_ParameterChangeCallbackHandle;
    std::unordered_map<std::string, std::unordered_set<IParameterChangeListener *>> m_ParameterChangeListeners;
    std::mutex m_MapAccessMutex;

public:
    // Derived nodes must wire SIGINT / custom interrupt behavior here.
    virtual void RegisterInterruptListener() = 0;
};
