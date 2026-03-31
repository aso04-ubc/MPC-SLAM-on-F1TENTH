#pragma once

#include "AppPCH.h"
#include "Utils/TypeUtils.hpp"
#include <type_traits>

std::shared_ptr<rclcpp::Node> CreateApplicationNode();

// Helpers for reading ROS 2 parameters and converting them to C++ types.
// When parameter types don't match the expected C++ type, an error can be
// logged (controlled by EnableParameterMismatchComplaints()).
namespace ParameterConversion {
    inline bool &GetComplainFlag() {
        static bool flag = false;
        return flag;
    }

    inline void EnableParameterMismatchComplaints(bool enable) {
        GetComplainFlag() = enable;
    }
}

namespace Impl {
    template<typename T>
    void Complain(const rclcpp::Parameter &parameter, const char *param_name) {
        // This is intentionally in a separate function so the noisy logging can
        // be turned off by the complain flag.
        if (ParameterConversion::GetComplainFlag()) {
            RCLCPP_ERROR(rclcpp::get_logger("ParameterConversion"),
                         "Parameter type mismatch for parameter '%s': "
                         "expect type to be convertible to C++ type '%s', but the "
                         "underlying parameter type is '%s'",
                         param_name,
                         Utils::PrettyTypeNameOf<T>().data(),
                         rclcpp::to_string(parameter.get_type()).c_str());
        }
    }

    template<typename T, typename Enable = void>
    struct ConvertParameterImpl {
        static std::optional<T> Convert(const rclcpp::Parameter &parameter, const char *name);
    };
}

namespace ParameterConversion {
    template<typename T>
    std::optional<T> Convert(const rclcpp::Parameter &parameter, const char *name = nullptr) {
        return Impl::ConvertParameterImpl<T>::Convert(parameter, name);
    }
}

namespace Impl {
    template<typename T>
    struct ConvertParameterImpl<T, typename std::enable_if<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value>::type> {
        static std::optional<T> Convert(const rclcpp::Parameter &parameter, const char *name) {
            if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
                return static_cast<T>(parameter.as_double());
            if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                return static_cast<T>(parameter.as_int());

            if (name) Complain<T>(parameter, name);
            return std::nullopt;
        }
    };

    template<>
    struct ConvertParameterImpl<bool, void> {
        static std::optional<bool> Convert(const rclcpp::Parameter &parameter, const char *name) {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
                if (name) Complain<bool>(parameter, name);
                return std::nullopt;
            }
            return parameter.as_bool();
        }
    };

    template<>
    struct ConvertParameterImpl<std::string, void> {
        static std::optional<std::string> Convert(const rclcpp::Parameter &parameter, const char *name) {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
                if (name) Complain<std::string>(parameter, name);
                return std::nullopt;
            }
            return parameter.as_string();
        }
    };

    template<typename T>
    struct ConvertParameterImpl<std::vector<T>, typename std::enable_if<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value>::type> {
        static std::optional<std::vector<T>> Convert(const rclcpp::Parameter &parameter, const char *name) {
            std::vector<T> result;
            if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
                auto arr = parameter.as_double_array();
                result.reserve(arr.size());
                for (double val: arr) result.push_back(static_cast<T>(val));
                return result;
            }
            if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
                auto arr = parameter.as_integer_array();
                result.reserve(arr.size());
                for (int64_t val: arr) result.push_back(static_cast<T>(val));
                return result;
            }
            if (name) Complain<std::vector<T>>(parameter, name);
            return std::nullopt;
        }
    };

    template<>
    struct ConvertParameterImpl<std::vector<std::string>, void> {
        static std::optional<std::vector<std::string>> Convert(const rclcpp::Parameter &parameter, const char *name) {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING_ARRAY) {
                if (name) Complain<std::vector<std::string>>(parameter, name);
                return std::nullopt;
            }
            return parameter.as_string_array();
        }
    };

    template<>
    struct ConvertParameterImpl<std::vector<bool>, void> {
        static std::optional<std::vector<bool>> Convert(const rclcpp::Parameter &parameter, const char *name) {
            if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
                if (name) Complain<std::vector<bool>>(parameter, name);
                return std::nullopt;
            }
            return parameter.as_bool_array();
        }
    };
}

namespace NodeUtils {
    template<typename T>
    std::optional<T> GetParameter(rclcpp::Node *node, const char *name) {
        // Declares the parameter if it doesn't exist yet, then converts it.
        if (!node->has_parameter(name))
            node->declare_parameter(name);
        return ParameterConversion::Convert<T>(node->get_parameter(name), name);
    }

    template<typename T>
    std::optional<T> GetParameter(rclcpp::Node *node, const char *name, const T& defaultValue) {
        // Same as above, but ensures a default value exists before conversion.
        if (!node->has_parameter(name)) {
            node->declare_parameter(name, defaultValue);
        }
        return ParameterConversion::Convert<T>(node->get_parameter(name), name);
    }
}
