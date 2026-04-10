#pragma once
#include <string_view>
#include <algorithm>
#include <string>
#include "AppPCH.h"

namespace Utils {
    // Small helper to represent a string literal as a compile-time type.
    // Used to enable constexpr slicing / concatenation without allocating.
    template<size_t N>
    struct StringLiteral {
        constexpr StringLiteral(const char (&str)[N]) {
            for (size_t i = 0; i < N; ++i) {
                Value[i] = str[i];
            }
        }

        char Value[N]{};

    public:
        constexpr StringLiteral() {
            Value[0] = '\0';
        }

        constexpr std::string_view View() const {
            return std::string_view(Value, N - 1);
        }

        constexpr size_t Size() const {
            return N - 1;
        }

        template<size_t Begin, size_t End>
        constexpr auto Slice() const {
            static_assert(Begin <= End, "Begin must be less than or equal to End");
            static_assert(End <= N - 1, "End must be less than or equal to the size of the string literal");
            StringLiteral<End - Begin + 1> result{};
            for (size_t i = Begin; i < End; ++i) {
                result.Value[i - Begin] = Value[i];
            }
            result.Value[End - Begin] = '\0';
            return result;
        }

        constexpr StringLiteral(const StringLiteral &other) {
            for (size_t i = 0; i < N; ++i) {
                Value[i] = other.Value[i];
            }
        }
    };


    template<size_t N>
    StringLiteral(const char (&)[N]) -> StringLiteral<N>;

    template<size_t N1, size_t N2>
    constexpr bool operator==(const StringLiteral<N1> &lhs, const StringLiteral<N2> &rhs) {
        return lhs.View() == rhs.View();
    }

    template<size_t N1, size_t N2>
    constexpr bool operator!=(const StringLiteral<N1> &lhs, const StringLiteral<N2> &rhs) {
        return !(lhs == rhs);
    }

    template<size_t N1, size_t N2>
    constexpr auto operator+(const StringLiteral<N1> &lhs, const StringLiteral<N2> &rhs) {
        StringLiteral<N1 + N2 - 1> result{};
        std::copy_n(lhs.Value, N1 - 1, result.Value);
        std::copy_n(rhs.Value, N2, result.Value + N1 - 1);
        return result;
    }
}

namespace Utils {
    // Produces a human-readable type name for debugging/logging.
    // Internally it extracts the template argument from __PRETTY_FUNCTION__.
    namespace Impl {
        template<typename T>
        inline const std::string &PrettyTypeNameOfImplFunc() {
            static const std::string name = []() {
                std::string_view pretty = __PRETTY_FUNCTION__;
                std::string_view begin_marker = "T = ";

                const size_t begin_pos = pretty.find(begin_marker);
                if (begin_pos == std::string_view::npos) {
                    return std::string(pretty);
                }

                const size_t start_pos = begin_pos + begin_marker.size();
                size_t end_pos = pretty.find_first_of(']', start_pos);
                end_pos = std::min(end_pos, pretty.find_first_of(';', start_pos));
                if (end_pos == std::string_view::npos || end_pos <= start_pos) {
                    end_pos = pretty.size();
                }

                return std::string(pretty.substr(start_pos, end_pos - start_pos));
            }();
            return name;
        }
    }

    template<typename T>
    inline std::string_view PrettyTypeNameOf() {
        return Impl::PrettyTypeNameOfImplFunc<T>();
    }
}
