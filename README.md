# Project B7

## Project Overview

This repository contains the final Project B7 ROS 2 workspace for our F1TENTH autonomous racing system. The active runtime stack combines a local reactive MPC controller with a safety / AEB node and custom prioritized control messages.

The main project goal is to drive from LiDAR-only local perception without requiring a pre-built global map. The final system extracts a local drivable corridor from the forward LiDAR scan, shapes a short-horizon reference inside that corridor, solves a local MPC problem, and then passes the command through a safety layer before publishing the final `/drive` command.

Optional and experimental code is also retained in this repository for comparison and project continuity, especially the `yolo/` folder used for offline perception experiments.

Note: the active launch file keeps the legacy name `milestone3_py.py` for repository continuity, but it is the final project launch entry currently used for the local MPC stack.

## Final Scope

The active project pipeline is:

- `mpc_controller`: local reactive MPC for nominal driving
- `safety`: command arbitration and LiDAR-based automatic emergency braking
- `dev_b7_interfaces`: custom control message definitions used between nodes
- `milestones/launch/milestone3_py.py`: current launch entry point for the project stack

The final system does not rely on a separate persistent global planner in the current runtime path. Instead, it performs local corridor extraction and local reference generation directly inside the MPC node.

In addition to the active local MPC runtime path, the broader final project also included a simulation-level global planning demonstration discussed in the final report. That global layer is not part of the default runtime launch documented in this README, and on-car global planning integration remained partial.

## System Architecture

### Active Runtime Stack

```text
/scan --------------------------+
                                |
                                v
                     +----------------------+
                     |    mpc_controller    |
                     | local corridor + MPC |
                     +----------------------+
                                |
                                | /drive_control
                                v
/scan --------------------------+------------------+
/odom or /ego_racecar/odom -----+                  |
                                                   v
                                      +----------------------+
                                      |        safety        |
                                      | arbitration + AEB    |
                                      +----------------------+
                                                   |
                                                   | /drive
                                                   v
                                                vehicle
```

### Node Roles

- `mpc_controller` subscribes to LiDAR and odometry, generates a local safe corridor, solves the short-horizon MPC problem, and publishes a prioritized `DriveControlMessage` on `/drive_control`.
- `safety` listens to `/drive_control`, keeps the highest-priority active command, evaluates obstacle risk using LiDAR and odometry, and publishes the final `AckermannDriveStamped` command on `/drive`.
- `dev_b7_interfaces` defines the prioritized command message used for controller-to-safety communication.

### Figure Placeholders

Placeholder: insert an updated system architecture figure here.

Placeholder: insert a ROS node / topic diagram here.

## Repository Structure

```text
.
|-- README.md
|-- CONTRIBS.md
|-- dev_b7_interfaces/   # Custom ROS 2 message definitions for prioritized control
|-- milestones/          # Launch package; current entry point is launch/milestone3_py.py
|-- mpc_controller/      # Local reactive MPC controller
|-- pic/                 # Images and diagrams used by documentation
|-- safety/              # C++ safety node with command arbitration and AEB
|-- yolo/                # Optional / experimental offline perception utilities
```

### Main Runtime Path

The active runtime chain for the project is:

1. `mpc_controller`
2. `safety`
3. `/drive` output to simulator or car

### Reference / Experimental Code

- `yolo/` contains offline scripts for semantic segmentation experiments, dataset preparation, and LiDAR-camera semantic association.
- The broader final project also included a simulation-level global planning demonstration described in the report; that layer is not part of the default launch path documented here.
- `pic/` currently contains older diagrams and can be updated with project-specific figures before submission.

## Dependencies

### ROS 2 Workspace Dependencies

This repository assumes a working ROS 2 environment with the standard packages used by the F1TENTH stack. The list below reflects the packages actually referenced by the current code and build files, even though some package manifests are still stale:

- `rclpy`
- `rclcpp`
- `ackermann_msgs`
- `nav_msgs`
- `sensor_msgs`
- `rosidl_default_generators`

### Python Dependencies

The active MPC node imports:

- `numpy`
- `scipy`
- `osqp`
- `opencv-python`

Install them manually if they are not already available in your environment:

```bash
pip install -r requirements.txt
```

### Optional Offline Dependencies

The optional `yolo/` utilities additionally use:

- `ultralytics==8.4.34`
- `rosbags`

These are not required for the main runtime pipeline.

### Build Dependencies for Safety

The `safety` package is a C++ package built with `ament_cmake`. It supports a CUDA path when a CUDA compiler is available, but it can also build without CUDA.

## Build And Run

### Build

From the workspace root:

```bash
colcon build --packages-select dev_b7_interfaces safety mpc_controller milestones
source install/setup.bash
```

Important note: `milestones/package.xml` still contains a legacy dependency on `wall_follow`, and some package metadata files are not yet fully synchronized with the active project stack. If your workspace does not provide that legacy package, you may need to clean up the manifest before the `milestones` package builds cleanly.

If you prefer to build the full workspace:

```bash
colcon build
source install/setup.bash
```

### Launch The Project Stack

The current launch entry point is:

```bash
ros2 launch milestones milestone3_py.py
```

Important note: although the launch file name is `milestone3_py.py`, it is the current launch file used by the project repository for the default local MPC runtime path.

### Simulator Versus Real Car

The current launch file hardcodes:

```python
sim = False
```

inside `milestones/launch/milestone3_py.py`. If you want to switch between simulator and real-car odometry topics, you currently need to edit that file directly or extend it with launch arguments.

### Expected Runtime Topics

- MPC input: `/scan`, `/odom` by default
- MPC output: `/drive_control`
- Safety input: `/drive_control`, `/scan`, `/odom` or `/ego_racecar/odom`
- Safety internal feedback topic: `/drive`
- Final vehicle command: `/drive`

## Core Nodes And Topics

### `mpc_controller`

File: `mpc_controller/mpc_controller/mpc_node.py`

Inputs:

- `/scan`
- `/odom`

Outputs:

- `/drive_control` as `dev_b7_interfaces/msg/DriveControlMessage`

Role in system:

- preprocess LiDAR data
- compute a gap-guided local target
- estimate left and right corridor bounds from forward LiDAR points
- build a short-horizon local MPC problem
- solve for speed and steering
- publish the nominal command with priority `1004`

### `safety`

Main implementation: `safety/safety_node/src/Core/DriveControlNode.cpp`

Inputs:

- `/drive_control`
- `/scan`
- `/odom` or `/ego_racecar/odom`
- `/drive` is also subscribed to internally by the LiDAR safety provider

Outputs:

- `/drive`

Role in system:

- maintain the latest active command for each priority
- forward the highest-priority command
- evaluate minimum distance and TTC using LiDAR-based safety providers
- override or reduce speed in critical and emergency states

### `dev_b7_interfaces`

Main interface file: `dev_b7_interfaces/msg/DriveControlMessage.msg`

Key fields:

- `priority`
- `active`
- `drive`

Role in system:

- allow multiple controllers or overrides to publish commands onto the same logical control channel
- let `safety` select the highest-priority active command

## Algorithm Overview

### Perception And Reference Generation

The project uses local LiDAR geometry rather than a global map. The controller first processes the forward scan to identify a navigable direction using a gap-following style heuristic. It then estimates left and right corridor boundaries from clustered forward LiDAR points and interpolates a dense local corridor over the MPC horizon.

That corridor is used in two ways:

- to define a nominal center / guided reference
- to impose safety-oriented lateral bounds inside the optimization problem

### MPC Controller

The controller models the vehicle with a short-horizon kinematic formulation and solves a local optimization problem at runtime. The optimization trades off:

- state tracking error
- terminal tracking error
- acceleration effort
- steering effort
- input rate smoothness
- slack penalty for soft constraint violation

The solved control output is a speed and steering command that respects configured bounds such as:

- horizon length
- control timestep
- wheelbase
- steering limit
- acceleration limit
- corridor safety margin

### Speed Shaping And Safety Override

Nominal speed is shaped based on local geometry such as target angle, curvature, corridor width, and forward clearance. The `safety` node then acts as the final authority on the output command.

If the local scene is safe, `safety` forwards the highest-priority controller command. If the time-to-collision or minimum distance becomes unsafe, it applies stage-based intervention ranging from conservative speed reduction to full emergency braking.

## Key Parameters

The most important parameters currently exposed inside `mpc_controller/mpc_controller/mpc_node.py` include:

### MPC Timing And Vehicle Model

- `dt`
- `horizon`
- `wheelbase`
- `control_rate_hz`

### State And Input Limits

- `max_speed`
- `min_speed`
- `max_accel`
- `min_accel`
- `max_steer`
- `max_ddelta`

### Corridor / Reference Generation

- `path_x_max`
- `path_y_limit`
- `corridor_bin_half_width`
- `corridor_margin`
- `corridor_min_half_width`
- `corridor_dense_points`
- `corridor_front_fov_deg`

### Gap Guidance

- `ftg_max_range`
- `ftg_min_safe_distance`
- `ftg_car_width`
- `goal_min_distance`
- `goal_max_distance`
- `lookahead_base`
- `lookahead_front_gain`

### Cost Weights

- `q_x`, `q_y`, `q_psi`, `q_v`
- `qf_x`, `qf_y`, `qf_psi`, `qf_v`
- `r_a`, `r_delta`
- `rd_a`, `rd_delta`
- `slack_weight`

### Safety Parameters

The launch file currently passes the following important parameters to `safety`:

- `ttc_full`
- `distance_full`
- `ttc_partial`
- `partial_brake_decel`
- `aeb_auto_release`

## Testing And Validation

This README focuses on code-specific documentation, so this section emphasizes repository-level validation and debugging workflow rather than full experimental discussion. The recommended workflow is:

1. Build the workspace and confirm all packages compile.
2. Run the project stack in simulation or on recorded data.
3. Visualize `/scan`, `/drive_control`, and `/drive` to confirm nominal control and safety intervention behavior.
4. Check that the car remains within corridor constraints and slows or stops appropriately when forward clearance collapses.

### Bag Replay

If you have a recorded ROS bag containing at least `/scan` and the expected odometry topic, you can use replay for controller and safety validation:

```bash
ros2 launch milestones milestone3_py.py
ros2 bag play <bag_path>
```

If replayed topics do not match the active configuration, first check:

- whether the bag publishes `/odom` or `/ego_racecar/odom`
- whether `sim` inside `milestones/launch/milestone3_py.py` matches the intended topic set
- whether the bag includes `/scan`

### Runtime Checks

The most useful runtime topics to inspect are:

- `/scan` for incoming LiDAR
- `/drive_control` for nominal controller output
- `/drive` for the final command after safety arbitration

To confirm that nominal control is reaching the safety layer, compare `/drive_control` and `/drive` during normal operation. To confirm AEB or safety override behavior, look for cases where `/drive_control` requests motion but `/drive` is reduced or forced toward zero speed by the safety node.

### Suggested Validation Cases

Suggested validation cases for this repository include:

- straight corridor tracking
- corner entry and exit stability
- narrow-gap behavior
- obstacle approach with AEB intervention
- simulator versus on-car topic configuration

Placeholder: insert testing screenshots, plots, or bag replay examples here.

## Known Limitations

- The current launch file uses a legacy name, `milestone3_py.py`, and does not yet expose launch arguments for `sim`.
- Some package metadata files are still out of sync with the active stack. For example, `milestones/package.xml` still declares `wall_follow`, while the current launch path uses `mpc_controller`, and several package descriptions remain placeholders.
- The runtime pipeline is local and reactive; it does not currently include a separate global mapping or global planning module.
- The `yolo/` code is experimental and offline-oriented rather than integrated into the active launch path.

## Submission Notes

This README is intended to document the codebase and how to run it. It is not meant to duplicate the full report. Background, literature review, long-form design discussion, and detailed experimental analysis should remain in the written report, while this file should stay focused on:

- repository structure
- dependencies
- build and launch workflow
- nodes, topics, and interfaces
- implementation-specific algorithm behavior
