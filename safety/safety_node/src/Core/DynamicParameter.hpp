#pragma once

#include "AppPCH.h"
#include "Utils/ParameterUtils.hpp"
#include "Core/NodeBase.hpp"
#include "Core/Interfaces.hpp"
#include "Utils/TypeUtils.hpp"

template<typename T>
class DynamicParameter : public IParameterChangeListener {
public:
    DynamicParameter() = delete;

    DynamicParameter(INodeExtend *node, const char *name, T defaultValue)
        : m_Node(node), m_ParameterName(name) {
        // Use GetParameter with default value to ensure parameter is declared with the default
        auto value = NodeUtils::GetParameter<T>(node, name, defaultValue);
        if (value) {
            m_CachedValue = boost::make_shared<const T>(std::move(*value));
        } else {
            // If still failed, fallback to default value
            m_CachedValue = boost::make_shared<const T>(std::move(defaultValue));
        }
        node->RegisterParameterChangeListener(name, this);
    }

public:
    boost::shared_ptr<const T> GetPtrNullable() const {
        return m_CachedValue.load();
    }

    T GetCopied() const {
        auto ptr = GetPtrNullable();
        if (ptr) {
            return *ptr;
        }
        throw std::runtime_error("Parameter value is not available");
    }

public:
    DynamicParameter(const DynamicParameter &) = delete;

    DynamicParameter(DynamicParameter &&) = delete;

    virtual ~DynamicParameter() override {
        if (m_Node) {
            m_Node->UnregisterParameterChangeListener(m_ParameterName, this);
        }
    }

protected:
    virtual void OnParameterChanged(const rclcpp::Parameter &parameter) override {
        auto newValue = ParameterConversion::Convert<T>(parameter, m_ParameterName.c_str());
        if (newValue) {
            m_CachedValue = boost::make_shared<const T>(std::move(*newValue));
        }
    }

private:
    INodeExtend *m_Node{nullptr};
    std::string m_ParameterName;
    boost::atomic_shared_ptr<const T> m_CachedValue{nullptr};
};
