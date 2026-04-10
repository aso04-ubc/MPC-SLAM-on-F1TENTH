# Real-Time MPC-Based Autonomous Racing with Obstacle Avoidance for F1TENTH

## Overview

This project implements a real-time autonomous racing stack for the [F1TENTH](https://f1tenth.org/) platform, built on **ROS 2 Foxy**. The core contribution is a **reactive local Model Predictive Controller (MPC)** that runs on the physical car using LiDAR-derived corridors, gap-based target selection, and a linearized kinematic bicycle model solved as a constrained quadratic program (QP) via OSQP.

In addition to the local controller, the project includes:

- A **global planning** layer demonstrated in simulation (and launchable on hardware), located on the **`sim_working_version`** branch.
- A **C++ safety node** providing time-to-collision (TTC) and distance-based automatic emergency braking (AEB).
- An **experimental YOLO-based semantic perception** pipeline (standalone scripts, **not** integrated into the ROS system).

![MPC without planner](pic/mpc%20without%20planner.gif)

---

## Getting Started

### Prerequisites

- **ROS 2 Foxy** (Ubuntu 20.04)
- F1TENTH simulation environment or physical vehicle
- Python 3.8+

### Build

```bash
# Source ros env first
source /opt/ros/foxy/setup.sh
# From the workspace root
cd ~/sim_ws
colcon build
source install/setup.bash
```

### Launch - Local Reactive MPC + Safety (main branch)

```bash
ros2 launch milestones mpc_start_up.py
```

This launches:
- `safety_node` - C++ AEB safety layer
- `mpc_controller_node` - Python MPC controller

The launch file defaults to real-car mode. To switch to simulation, edit `milestones/launch/mpc_start_up.py` and set `sim = True`.

### Launch - Global Planner in Simulation (sim_working_version branch)

The global planner that successfully runs in simulation lives on the **`sim_working_version`** branch. Use the following instructions to run it:

```bash
# Switch to the sim_working_version branch
cd src/Project_B7
git checkout sim_working_version

# Rebuild
cd ~/sim_ws
colcon build --packages-select dev_b7_interfaces safety mpc_controller milestones
source install/setup.bash

# Launch the global planning stack in simulation
ros2 launch milestones race_line_stack.launch.py sim:=true odom_topic:=/ego_racecar/odom map_window_size:=1000
```

For best performance, replace the default Levine map with `./levine.png` in the launch configuration.

Adjust `map_window_size` based on the resolution of your simulation map. The global planner is **computationally intensive** - the larger the map, the more resources are required. As a reference, a 1000 px map requires approximately **16 CPU cores** and **10 GB of memory** to run successfully.


### ROS Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Input | LiDAR scan data |
| `/odom` | `nav_msgs/Odometry` | Input | Vehicle odometry |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Output | Final drive commands (via safety node) |
| `/drive_control` | `dev_b7_interfaces/DriveControlMessage` | Internal | Priority-based drive commands from controller to safety node |

---

## System Architecture

The deployed system has two layers plus a standalone perception experiment:

| Layer | Description | Hardware Status |
|---|---|---|
| **Reactive Local MPC** | Consumes LiDAR geometry + odometry, solves a constrained QP to produce steering and speed commands at 30 Hz. | Fully deployed on car |
| **Global Planning** | Converts an occupancy-grid map into a race-line-optimized global path with a curvature-limited speed profile. | Demonstrated in simulation; launches on hardware but limited by IMU drift |
| **YOLO Perception** | YOLOv8-seg labels camera images and projects semantics onto LiDAR space for wall-vs-obstacle classification. | Standalone scripts only, not integrated into ROS |

![MPC with planner](pic/mpc%20with%20planner.gif)

---

## Repository Structure

```
Project_B7/
|-- mpc_controller/            # ROS 2 Python package - local reactive MPC
|   |-- mpc_controller/
|   |   |-- mpc_node.py        # Main MPC node
|   |   |-- gap_utils.py       # Gap-follow algorithm for target selection
|-- safety/                    # ROS 2 C++ package - AEB / safety layer
|   |-- safety_node/src/
|   |   |-- Core/              # DriveControlNode, Executor, SafetyInfo, etc.
|   |   |-- Providers/         # LidarSafetyInfoProvider, DepthMapSafetyInfoProvider
|   |   |-- Utils/             # Parameter and type utilities
|-- dev_b7_interfaces/         # Custom ROS 2 message definitions
|   |-- msg/
|   |   |-- DriveControlMessage.msg
|   |   |-- ControlSubmissionMessage.msg
|-- milestones/                # Launch files
|   |-- launch/
|   |   |-- mpc_start_up.py    # Launches safety_node + mpc_controller_node
|-- yolo/                      # Experimental YOLO perception (standalone, NOT in ROS)
|   |-- yolo.py                # Semantic segmentation + depth overlay visualization
|   |-- yolo laser match.py    # LiDAR-to-camera column mapping with semantic labels
|   |-- extract pics.py        # Extract training images from ROS bags
|   |-- best.pt                # Trained YOLOv8-seg model weights
|   |-- README.md
|-- pic/                       # Demo GIFs and figures
|   |-- mpc without planner.gif
|   |-- mpc with planner.gif
|   |-- global planner.gif
|   |-- yolo detaction.gif
|   |-- yolo laser match.gif
|   |-- on car map builder.gif
|-- CONTRIBS.md
|-- README.md                  # This file
```

### Branches

| Branch | Purpose |
|---|---|
| `main` | Primary branch - local reactive MPC + safety node |
| **`sim_working_version`** | **Global planner that successfully runs in simulation** - contains the full planning stack with race-line optimization and speed profiling |
| `yolo` | YOLO development branch |
| `leo/planner` | Planner research branch |
| `offline-slam` | Offline SLAM mapping experiments |

---

## Reactive Local MPC Controller

The core of the project. Located in `mpc_controller/mpc_controller/mpc_node.py`.

### Kinematic Bicycle Model

The MPC uses a linearized kinematic bicycle model with state $z = [x, y, \psi, v]$ and control $u = [a, \delta]$:

$$
\begin{aligned}
\dot{x} &= v\cos(\psi) \\
\dot{y} &= v\sin(\psi) \\
\dot{\psi} &= \frac{v}{L}\tan(\delta) \\
\dot{v} &= a
\end{aligned}
$$

where $L = 0.50\ \mathrm{m}$ is the wheelbase. The model is linearized around a nominal reference at each time step and discretized for the QP formulation:

$$
z_{k+1} \approx A_k z_k + B_k u_k + g_k
$$

### Optimization Objective

The QP minimizes a cost function penalizing state-tracking error, control effort, input-rate variation, and corridor constraint violations (via slack):

$$
\min \sum_k \|z_k - z_k^{\mathrm{ref}}\|_Q^2 + \sum_k \|u_k - u_k^{\mathrm{ref}}\|_R^2 + \sum_k \|u_k - u_{k-1}\|_{R_d}^2 + w_s \sum_k s_k^2
$$

subject to steering limits, acceleration bounds, speed bounds, heading constraints, and soft corridor boundary constraints.

### Key Design Choices

- **Linearized (not fully nonlinear) MPC**: Reduces OSQP solve time from about 20-50 ms to about 2-10 ms, critical for stable 30 Hz control on a small vehicle.
- **LiDAR-first local control**: Avoids dependence on global localization, making it robust on hardware.
- **Blended reference generation**: Combines corridor centerline, gap direction, outside-bias heuristics, and terminal goal blending - pure centerline following is too timid, pure gap-follow is too reactive.
- **Rate-limited steering**: Maximum steering change of 0.024 rad between consecutive commands to reduce oscillation.

### Representative Parameters

| Parameter | Value | Role |
|---|---|---|
| Control rate | 30 Hz | Closed-loop command frequency |
| Horizon N | 11 | Short predictive horizon |
| Time step $\Delta t$ | 0.06 s | Discretization interval |
| Wheelbase | 0.50 m | Kinematic bicycle geometry |
| Max steering | 0.36 rad | Steering constraint |
| Max steering step | 0.024 rad | Rate-limiting for smoothness |
| Speed upper bound | 4.0 m/s | Velocity constraint |
| Solver | OSQP | Quadratic program solver |

![mpc without planner](pic/mpc%20without%20planner.gif)
---

## LiDAR Corridor Extraction and Gap-Guided Reference

Located in `mpc_controller/mpc_controller/gap_utils.py` and the corridor-extraction logic within `mpc_node.py`.

The pipeline:

1. **Gap-follow module** identifies a forward gap from LiDAR ranges, proposing a goal angle and safe distance. Uses disparity extension and cost-weighted beam selection.
2. **Goal filtering** smooths the target across frames with exponential filters and lateral deadbanding to avoid abrupt jumps.
3. **Corridor extraction** transforms forward-facing LiDAR points into a local Cartesian frame, clusters them into left/right sides, and reconstructs dense boundary curves by binning and smoothing.
4. **Reference shaping** blends corridor centerline, gap direction, an outside-bias heuristic (delays turn-in for smoother corner entry), and a terminal blend toward the filtered goal.

---

## Safety Node

Located in `safety/`. A C++ ROS 2 node that acts as the final gatekeeper before drive commands reach the vehicle.

- Uses **time-to-collision (TTC)** and **minimum obstacle distance** from LiDAR (and optionally depth camera) to trigger multi-stage braking responses.
- Configurable TTC and distance thresholds with dynamic parameter support.
- Implements a priority-based drive command arbitration system via custom `DriveControlMessage` messages.
- Sends an emergency stop on node shutdown.

---

## Global Planner (sim_working_version branch)

The full global planning stack is located on the **`sim_working_version`** branch and has been demonstrated working in simulation. See [Getting Started](#launch--global-planner-in-simulation-sim_working_version-branch) for launch instructions.

### How It Works

1. **Map processing**: Converts an occupancy-grid map into a drivable binary mask and extracts a closed centerline.
2. **Race-line optimization**: Optimizes lateral offsets from the centerline, minimizing curvature and lateral variation subject to free-space bounds. Solved with L-BFGS-B (SciPy):

   $$
   \mathbf{r}_i = \mathbf{c}_i + e_i \mathbf{n}_i
   $$

   $$
   \min_{\mathbf{e}} \; w_\kappa \sum_i (\Delta^2 e_i)^2 + w_s \sum_i (\Delta e_i)^2 + w_c \sum_i e_i^2 \quad \text{s.t. } -b_i \le e_i \le b_i
   $$


3. **Speed profiling**: Generates a curvature-limited speed profile with lateral-acceleration caps and forward/backward longitudinal-feasibility passes.

![Global planner demo](pic/global%20planner.gif)

### Hardware Status

The global-planning stack launches on the physical car, confirming correct node interfaces and execution flow. However, on-car performance is limited because:

- **IMU noise and odometry drift** degrade localization quality.
- The global layer is far more sensitive to pose accuracy than the reactive local controller.
- This identifies **state estimation** as the primary bottleneck for robust hardware-level global planning.

![On-car map building](pic/on%20car%20map%20builder.gif)
---

## Experimental YOLO Perception

All YOLO-related code is located in the **`yolo/`** folder. This is a **standalone experiment** that was **not integrated into the ROS 2 system** and serves as a preliminary exploration of semantic perception for the F1TENTH platform.

![YOLO detection](pic/yolo%20detaction.gif)

### What It Does

- **`yolo.py`**: Runs YOLOv8-seg inference on ROS bag camera data, overlays semantic segmentation masks (road, wall, obstacle) on RGB images, and queries aligned depth maps for distance estimation.
- **`yolo laser match.py`**: Projects YOLO semantic labels into LiDAR space using camera-to-LiDAR column mapping. Each LiDAR point inherits the semantic class of its corresponding image column, producing a semantically colored bird's-eye-view LiDAR visualization.
- **`extract pics.py`**: Utility to extract random camera frames from ROS bags for training data collection.
- **`best.pt`**: The best-performing YOLOv8-seg model trained on our custom dataset.

![YOLO-LiDAR matching](pic/yolo%20laser%20match.gif)

### Training Pipeline

1. Manually annotated an initial seed set of **30 images** using [Label Studio](https://labelstud.io/).
2. Trained an initial YOLOv8-seg model on the seed set.
3. Connected the trained model back to Label Studio for **pre-labeling** additional images.
4. Manually corrected and refined machine-generated annotations, then added them back to the dataset.
5. Iterated this human-in-the-loop bootstrapping process to expand the labeled data efficiently.

Training data and additional artifacts are available on [Google Drive](https://drive.google.com/drive/folders/1-zE6DV8pEdiYcfC7sHqsTIyIuEYDFliM?usp=sharing).

### Why It Was Not Deployed

The F1TENTH vehicle does **not** have CUDA-capable hardware. CPU-only YOLOv8 inference required ~500 ms per frame, far below the real-time requirement for closed-loop control. The experiment remains a proof-of-concept for semantic wall-vs-obstacle labeling.

### Running YOLO Scripts

```bash
cd yolo/

# Install dependencies (no ROS environment required)
pip install ultralytics==8.4.34 rosbags

# Visualize YOLO segmentation + depth overlay
python yolo.py

# Visualize LiDAR-to-camera semantic column mapping
python "yolo laser match.py"

# Extract training images from a ROS bag
python "extract pics.py"
```

Place your ROS bag files in the `yolo/` directory (or modify `BAG_PATH` in the scripts).

---

## Dependencies

### ROS 2 Packages

- `ackermann_msgs`
- `nav_msgs`
- `sensor_msgs`
- `std_msgs`
- `rclpy` / `rclcpp`

### Python (MPC Controller)

- `numpy`
- `scipy`
- `osqp`
- `opencv-python` (for debug visualization)

### Python (YOLO - standalone, no ROS required)

- `ultralytics==8.4.34`
- `rosbags`
- `opencv-python`
- `numpy`

### C++ (Safety Node)

- Standard ROS 2 C++ build toolchain (`ament_cmake`)

---

## Project Status

| Subsystem | Status | Summary |
|---|---|---|
| Local reactive MPC on car | **Achieved** | Closed-loop local control runs on the physical F1TENTH platform with stable steering and speed commands from LiDAR-driven local geometry. |
| Global planning in simulation | **Achieved** | Track-level planning demonstrated successfully in simulation (see `sim_working_version` branch). |
| Global planning on car | **Partial** | Stack launches on hardware, but trajectory quality is limited by IMU and localization errors. |
| Experimental YOLO perception | **Experimental** | YOLO-based labels tested and projected into LiDAR space. Standalone only - not integrated into ROS. |

### Key Limitations

- **Hardware localization**: IMU noise, odometry drift, and calibration errors degrade global planner performance on the real car. State estimation is the primary bottleneck.
- **No CUDA on vehicle**: Prevents real-time YOLO inference, limiting perception to an offline experiment.
- **Qualitative evaluation**: The project prioritized a stable demonstration and clear architectural understanding over a comprehensive quantitative benchmark suite.
