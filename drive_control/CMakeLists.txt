cmake_minimum_required(VERSION 3.5)
project(safety)


# Default to C++
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



find_package(rclcpp REQUIRED)
find_package(dev_b7_interfaces REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

include(FetchContent)
FetchContent_Declare(argparse
        GIT_REPOSITORY
        https://github.com/p-ranav/argparse
        GIT_TAG
        v3.2)
FetchContent_MakeAvailable(argparse)

SET(DRIVE_CONTROL_NODE_NAME safety_node)

add_executable(${DRIVE_CONTROL_NODE_NAME}
        drive_control_node/src/Main.cpp
        drive_control_node/src/Application.hpp
        drive_control_node/src/NodeImpl.cpp
        drive_control_node/src/AppPCH.h
        drive_control_node/src/ExecutionContext.hpp
        drive_control_node/src/ExecutionContext.cpp)

ament_target_dependencies(
        ${DRIVE_CONTROL_NODE_NAME}
        rclcpp
        dev_b7_interfaces
        ackermann_msgs
        nav_msgs
        sensor_msgs
        rclcpp_lifecycle
)

ament_package()

target_link_libraries(
        ${DRIVE_CONTROL_NODE_NAME}
        argparse::argparse
)

target_precompile_headers(
        ${DRIVE_CONTROL_NODE_NAME}
        PRIVATE
        drive_control_node/src/AppPCH.h
)

target_compile_options(${DRIVE_CONTROL_NODE_NAME}
        PRIVATE
        -fconcepts
        -Wno-pedantic
)

install(TARGETS
        ${DRIVE_CONTROL_NODE_NAME}
        DESTINATION lib/${PROJECT_NAME}
)