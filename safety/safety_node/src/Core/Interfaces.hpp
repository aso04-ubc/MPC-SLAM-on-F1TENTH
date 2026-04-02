#pragma once

#include "AppPCH.h"

// Listener interface for dynamically changing ROS parameters.
// Implementations are notified when a subscribed parameter is updated.
class IParameterChangeListener {
public:
    virtual ~IParameterChangeListener() = default;

    // Called when a parameter with the listener's name is set/updated.
    virtual void OnParameterChanged(const rclcpp::Parameter &parameter) = 0;
};
