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

SET(DRIVE_CONTROL_NODE_NAME safety_node)

add_executable(${DRIVE_CONTROL_NODE_NAME}
        safety_node/src/Main.cpp
        safety_node/src/Application.hpp
        safety_node/src/NodeImpl.cpp
        safety_node/src/AppPCH.h
        safety_node/src/ExecutionContext.hpp
        safety_node/src/ExecutionContext.cpp)

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

target_precompile_headers(
        ${DRIVE_CONTROL_NODE_NAME}
        PRIVATE
        safety_node/src/AppPCH.h
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
