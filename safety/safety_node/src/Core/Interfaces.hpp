#pragma once

#include "AppPCH.h"

class IParameterChangeListener {
public:
    virtual ~IParameterChangeListener() = default;

    virtual void OnParameterChanged(const rclcpp::Parameter &parameter) = 0;
};
