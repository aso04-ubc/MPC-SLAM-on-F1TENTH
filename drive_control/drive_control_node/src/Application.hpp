/**
 * @file Application.hpp
 * @brief Main application interface for the Drive Control Node
 *
 * This file defines the factory function and configuration structure for creating
 * the drive control node, which handles autonomous emergency braking (AEB) and
 * drive command arbitration.
 */

#pragma once

#include "AppPCH.h"
#include "ExecutionContext.hpp"

/**
 * @struct NodeCreationInfo
 * @brief Configuration parameters for creating the Drive Control Node
 *
 * This structure holds the configuration values needed to initialize the
 * Autonomous Emergency Braking (AEB) system. These parameters control when
 * the AEB system should engage to prevent collisions.
 */
struct NodeCreationInfo {
    /**
     * @brief Time-To-Collision (TTC) threshold for AEB activation
     *
     * The AEB system will engage if the calculated time until collision
     * with an obstacle is less than this value (in seconds).
     * Default: 0.8 seconds
     */
    double aeb_ttc_threshold = 0.3;

    /**
     * @brief Minimum distance threshold for AEB activation
     *
     * The AEB system will engage if any obstacle is closer than this
     * distance (in meters), regardless of TTC.
     * Default: 0.3 meters
     */
    double aeb_minimum_distance = 0.5;
};

/**
 * @brief Factory function to create the Drive Control Node
 *
 * This function creates and returns a shared pointer to a lifecycle node
 * that manages drive control, AEB, and command arbitration.
 *
 * @param creation_info Configuration parameters for AEB thresholds
 * @return std::shared_ptr<ILifecycleNodeEXT> Pointer to the created node
 */
std::shared_ptr<ILifecycleNodeEXT> CreateApplicationNode(const NodeCreationInfo& creation_info);
