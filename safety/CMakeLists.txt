cmake_minimum_required(VERSION 3.16)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)
set(CMAKE_ERROR_DEPRECATED OFF CACHE BOOL "" FORCE)

if (NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER nvcc
            PATHS /usr/local/cuda-11.4/bin /usr/local/cuda-13.1/bin /usr/local/cuda/bin /usr/bin
            NO_DEFAULT_PATH)
    if (NOT CMAKE_CUDA_COMPILER)
        find_program(CMAKE_CUDA_COMPILER nvcc)
    endif ()
endif ()

if (CMAKE_CUDA_COMPILER AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Detect GPU architecture using nvidia-smi before enabling CUDA
    execute_process(
        COMMAND bash -c "nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'"
        OUTPUT_VARIABLE DETECTED_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if (DETECTED_ARCH)
        message(STATUS "Detected GPU architecture: ${DETECTED_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCH} CACHE STRING "CUDA architectures" FORCE)
    else()
        message(STATUS "Failed to detect GPU architecture, use device native.")
        # set(CMAKE_CUDA_COMPILER "")
        set(USE_NATIVE_ARCHITECHTURE ON)
    endif()
endif()

if (USE_NATIVE_ARCHITECHTURE)
    project(safety LANGUAGES CXX CUDA)
else()
    project(safety LANGUAGES CXX)
endif()

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(HAS_CUDA TRUE)
else()
    set(HAS_CUDA FALSE)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(dev_b7_interfaces REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

set(NODE_NAME safety_node)

set(SOURCES
        safety_node/src/Main.cpp
        safety_node/src/Core/DriveControlNode.cpp
        safety_node/src/Providers/LidarSafetyInfoProvider.cpp
        safety_node/src/Core/Executor.hpp
        safety_node/src/Utils/TypeUtils.hpp
        safety_node/src/Utils/ParameterUtils.hpp
        safety_node/src/Core/SafetyInfo.hpp
        safety_node/src/Providers/LidarSafetyInfoProvider.hpp
        safety_node/src/Providers/ComputeWrapper.hpp
        safety_node/src/AppPCH.h
        safety_node/src/Providers/ComputeWrapper.cpp
        safety_node/src/Providers/DepthMapSafetyInfoProvider.hpp
        safety_node/src/Providers/DepthMapSafetyInfoProvider.cpp
)

if (HAS_CUDA)
    list(APPEND SOURCES
            safety_node/src/Providers/DeviceComputation.cu
            safety_node/src/Providers/DeviceComputation.cuh
    )
endif ()

add_executable(${NODE_NAME} ${SOURCES})

set_target_properties(${NODE_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

if (HAS_CUDA AND NOT USE_NATIVE_ARCHITECHTURE)
    set_target_properties(${NODE_NAME} PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )

    target_compile_options(${NODE_NAME} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                -std=c++17
                --expt-relaxed-constexpr
                --generate-code=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=[compute_${CMAKE_CUDA_ARCHITECTURES},sm_${CMAKE_CUDA_ARCHITECTURES}]
            >
    )

    target_compile_definitions(${NODE_NAME} PUBLIC APP_USE_CUDA)

    find_package(CUDA REQUIRED)
    target_include_directories(${NODE_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${NODE_NAME} ${CUDA_LIBRARIES})
endif ()

ament_target_dependencies(${NODE_NAME}
        rclcpp
        dev_b7_interfaces
        ackermann_msgs
        nav_msgs
        sensor_msgs
)

target_include_directories(
        ${NODE_NAME}
        PRIVATE
        safety_node/src)

target_precompile_headers(${NODE_NAME}
        PRIVATE
        safety_node/src/AppPCH.h
)

install(TARGETS ${NODE_NAME}
        DESTINATION lib/${PROJECT_NAME}
)

ament_package()
