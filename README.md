# Real-Time MPC-Based Autonomous Racing with Obstacle Avoidance for F1TENTH

## Overview

This repository contains our final `Project_B7` ROS 2 stack for autonomous F1TENTH racing. The active runtime path on `main` is a **LiDAR-driven reactive local MPC controller** feeding a **C++ safety / AEB node** through custom prioritized control messages.

The primary design goal is to drive from **local perception only**, without requiring a pre-built global map during the default runtime path. The controller extracts a local drivable corridor from forward LiDAR data, shapes a short-horizon reference, solves a constrained quadratic program (QP) with OSQP, and passes commands through the safety layer before publishing the final `/drive` command.

In addition to the active local controller, the repository also includes:

- A **global planning** stack that runs successfully in simulation on the `sim_working_version` branch.
- A **C++ safety node** with time-to-collision (TTC) and distance-based automatic emergency braking (AEB).
- An **experimental YOLO-based semantic perception** pipeline kept as standalone scripts and not integrated into the ROS runtime stack.

![MPC without planner](pic/mpc%20without%20planner.gif)

---

## Getting Started

### Prerequisites

- **ROS 2 Foxy** on Ubuntu 20.04
- Python 3.8+
- F1TENTH simulator or physical vehicle setup

### Build

```bash
source /opt/ros/foxy/setup.sh
cd ~/sim_ws
pip install -r src/Project_B7/requirements.txt
colcon build --packages-select dev_b7_interfaces safety mpc_controller milestones
source install/setup.bash
```

If you prefer to build the full workspace instead of selected packages:

```bash
colcon build
source install/setup.bash
```

### Launch - Local Reactive MPC + Safety (`main`)

```bash
ros2 launch milestones mpc_start_up.py
```

This launch file starts:

- `safety_node` for arbitration and emergency braking
- `mpc_controller_node` for local corridor generation and MPC control

The launch file currently defaults to real-car mode. To switch to simulation, edit `milestones/launch/mpc_start_up.py` and set:

```python
sim = True
```

### Launch - Global Planner in Simulation (`sim_working_version`)

The global planning stack that successfully runs in simulation lives on the `sim_working_version` branch:

```bash
cd ~/sim_ws/src/Project_B7
git checkout sim_working_version

cd ~/sim_ws
colcon build --packages-select dev_b7_interfaces safety mpc_controller milestones
source install/setup.bash

ros2 launch milestones race_line_stack.launch.py sim:=true odom_topic:=/ego_racecar/odom map_window_size:=1000
```

For best performance, replace the default Levine map with `./levine.png` in that branch's launch configuration. The planner is computationally expensive; with a 1000 px map, a practical reference point is roughly **16 CPU cores** and **10 GB of memory**.

### Core ROS Topics

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | Input | LiDAR scan data |
| `/odom` | `nav_msgs/Odometry` | Input | Vehicle odometry for local MPC |
| `/drive_control` | `dev_b7_interfaces/DriveControlMessage` | Internal | Prioritized controller output sent to safety |
| `/drive` | `ackermann_msgs/AckermannDriveStamped` | Output | Final command after safety arbitration |

---

## System Architecture

The deployed system combines an active local runtime stack, a simulation-only global planner branch, and a standalone perception experiment.

### Active Runtime Stack

Without global planner we have:
![w/o control flow](pic/control%20flow%20fallback.png)

With global planner we have:
![w control flow](pic/control%20flow.png)

### Runtime Roles

- `mpc_controller` subscribes to LiDAR and odometry, builds a safe local corridor, solves the short-horizon MPC problem, and publishes a prioritized `DriveControlMessage`.
- `safety` listens to `/drive_control`, evaluates TTC and obstacle distance, forwards the highest-priority safe command, and publishes the final `/drive`.
- `dev_b7_interfaces` defines the prioritized command interface used between controller and safety node.

### High-Level Project Layers

| Layer | Description | Status |
|---|---|---|
| **Reactive Local MPC** | LiDAR corridor extraction + gap-guided reference + constrained QP control at 30 Hz. | Fully deployed on car |
| **Global Planning** | Occupancy-grid processing, race-line optimization, and speed profiling. | Demonstrated in simulation |
| **YOLO Perception** | Semantic camera/LiDAR experiments for wall-vs-obstacle classification. | Standalone only |

![MPC with planner](pic/mpc%20with%20planner.gif)

---

## Repository Structure

```text
Project_B7/
|-- README.md
|-- CONTRIBS.md
|-- requirements.txt
|-- dev_b7_interfaces/         # Custom ROS 2 message definitions
|-- milestones/
|   |-- launch/
|   |   |-- mpc_start_up.py    # Main local MPC + safety launch file
|-- mpc_controller/
|   |-- mpc_controller/
|   |   |-- mpc_node.py        # Main MPC node
|   |   |-- gap_utils.py       # Gap-follow target selection
|-- safety/                    # C++ safety / AEB package
|-- yolo/                      # Standalone perception experiments
|-- pic/                       # Demo GIFs and figures
```

### Branches

| Branch | Purpose |
|---|---|
| `main` | Primary branch with local reactive MPC + safety |
| `sim_working_version` | Simulation-capable global planning stack |
| `yolo` | YOLO development branch |
| `leo/planner` | Planner research branch |
| `offline-slam` | Offline SLAM and mapping experiments |

---

## Reactive Local MPC Controller

The core implementation lives in `mpc_controller/mpc_controller/mpc_node.py`.

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

where $L = 0.50\ \mathrm{m}$ is the wheelbase. Around a nominal reference, the system is linearized and discretized into:

$$
z_{k+1} \approx A_k z_k + B_k u_k + g_k
$$

### Optimization Objective

The QP penalizes state-tracking error, control effort, input-rate variation, and soft corridor-boundary violations:

$$
\min \sum_k \|z_k - z_k^{\mathrm{ref}}\|_Q^2 + \sum_k \|u_k - u_k^{\mathrm{ref}}\|_R^2 + \sum_k \|u_k - u_{k-1}\|_{R_d}^2 + w_s \sum_k s_k^2
$$

subject to steering, acceleration, speed, heading, and corridor constraints.

### Key Design Choices

- **Linearized MPC** keeps OSQP solve times low enough for stable real-time control.
- **LiDAR-first local control** avoids dependence on global localization during the main runtime path.
- **Blended reference generation** combines corridor centerline, gap direction, outside-bias heuristics, and terminal goal blending.
- **Rate-limited steering** reduces oscillation and improves vehicle stability.

### Representative Parameters

| Parameter | Value | Role |
|---|---|---|
| Control rate | 30 Hz | Closed-loop command frequency |
| Horizon `N` | 11 | Predictive horizon length |
| Time step `dt` | 0.06 s | Discretization interval |
| Wheelbase | 0.50 m | Vehicle geometry |
| Max steering | 0.36 rad | Steering bound |
| Max steering step | 0.024 rad | Steering rate limit |
| Speed upper bound | 4.0 m/s | Velocity constraint |
| Solver | OSQP | Quadratic program solver |

---

## LiDAR Corridor Extraction and Gap-Guided Reference

The local-reference pipeline is implemented across `mpc_controller/mpc_controller/gap_utils.py` and the corridor logic inside `mpc_node.py`.

### Pipeline Summary

1. **Gap-follow target selection** identifies a safe forward direction from LiDAR data.
2. **Goal filtering** smooths the selected target across frames to reduce abrupt turns.
3. **Corridor extraction** clusters forward LiDAR points into left and right boundaries and reconstructs a dense local corridor.
4. **Reference shaping** blends corridor centerline information with gap guidance and terminal goal biasing.

### Why This Matters

The controller uses the local corridor in two ways:

- to define a nominal short-horizon reference trajectory
- to impose soft lateral safety bounds inside the optimization problem

![mpc without planner](pic/mpc%20without%20planner.gif)

---

## Safety Node

The safety system is implemented in `safety/` and acts as the final gatekeeper before commands reach the vehicle.

### Main Responsibilities

- compute **time-to-collision (TTC)** and minimum obstacle distance from LiDAR
- arbitrate between prioritized control messages
- reduce or override speed in partial-brake and full-brake conditions
- publish an emergency-safe command on shutdown

### Interfaces

- Input topics: `/drive_control`, `/scan`, and odometry (`/odom` or `/ego_racecar/odom`)
- Output topic: `/drive`
- Main interface message: `dev_b7_interfaces/msg/DriveControlMessage`

---

## Global Planner (`sim_working_version`)

The full global planning stack is maintained on the `sim_working_version` branch and has been demonstrated successfully in simulation.

### Planner Flow

1. **Map processing** converts an occupancy-grid map into a drivable mask and extracts a closed centerline.
2. **Race-line optimization** solves for lateral offsets that reduce curvature and smoothness cost while respecting free-space bounds.
3. **Speed profiling** computes a curvature-limited speed profile with feasibility passes.

### Path-Planning Algorithm

Let $\mathbf{c}_i \in \mathbb{R}^2$ be sampled centerline points and $\mathbf{n}_i$ their unit normals. The race line is parameterized as:

$$
\mathbf{r}_i = \mathbf{c}_i + e_i \mathbf{n}_i,
$$

where $e_i$ is the lateral offset at sample $i$. Offsets are solved by bounded optimization:

$$
\min_{\mathbf{e}} \; J(\mathbf{e}) =
w_{\kappa}\sum_i (\Delta^2 e_i)^2 +
w_s\sum_i (\Delta e_i)^2 +
w_c\sum_i e_i^2,
\quad
\text{s.t. } -b_i \le e_i \le b_i,
$$

with $b_i$ estimated from local free-space distance (distance transform on the drivable mask). The first term penalizes curvature-like variation, the second enforces smooth offset transitions, and the third keeps the solution near the centerline when not required to deviate.

We solve this problem with **L-BFGS-B** (`scipy.optimize.minimize`) because the objective is smooth and differentiable, while each offset has simple box constraints ($-b_i \le e_i \le b_i$). This method is computationally efficient for medium-sized waypoint vectors, handles per-variable bounds directly, and is robust for repeated replanning without requiring manual constraint projection.

After geometry optimization, speed is limited by curvature and vehicle acceleration bounds:

$$
v_i^{\text{lat}} = \sqrt{\frac{a_{\text{lat,max}}}{\max(|\kappa_i|,\epsilon)}}, \qquad
v_i \le v_{\max},
$$

followed by forward/backward passes to enforce longitudinal acceleration and braking feasibility along arc length.

![Global planner demo](pic/global%20planner.gif)

### Hardware Status

The planner launches on hardware, but robust on-car performance is limited by localization quality:

- IMU noise and odometry drift degrade state estimation
- the global layer is more sensitive to pose quality than the local reactive controller
- state estimation remains the primary bottleneck for dependable hardware-level global planning

![On-car map building](pic/on%20car%20map%20builder.gif)

---

## Experimental YOLO Perception

All YOLO-related code lives in `yolo/`. This part of the repository is a standalone experiment and was not integrated into the ROS 2 runtime stack.

### Included Scripts

- `yolo.py`: YOLOv8-seg inference plus depth-overlay visualization
- `yolo laser match.py`: semantic projection from camera space into LiDAR columns
- `extract pics.py`: random image extraction from ROS bags for data collection
- `best.pt`: trained YOLOv8-seg weights

![YOLO detection](pic/yolo%20detaction.gif)
![YOLO-LiDAR matching](pic/yolo%20laser%20match.gif)

### Training Workflow

1. Label an initial seed set of images.
2. Train an initial YOLOv8-seg model.
3. Reuse the model for pre-labeling more data.
4. Correct generated labels manually.
5. Iterate to grow the dataset efficiently.

Training artifacts are available on [Google Drive](https://drive.google.com/drive/folders/1-zE6DV8pEdiYcfC7sHqsTIyIuEYDFliM?usp=sharing).

### Why It Was Not Deployed

The F1TENTH vehicle does not have CUDA-capable hardware, and CPU-only YOLO inference was too slow for real-time closed-loop use.

### Running YOLO Scripts

```bash
cd yolo
pip install ultralytics==8.4.34 rosbags
python yolo.py
python "yolo laser match.py"
python "extract pics.py"
```

---

## Dependencies

### ROS 2 Packages

- `rclpy`
- `rclcpp`
- `ackermann_msgs`
- `nav_msgs`
- `sensor_msgs`
- `rosidl_default_generators`

### Python Dependencies

- `numpy`
- `scipy`
- `osqp`
- `opencv-python`

### Optional Offline Dependencies

- `ultralytics==8.4.34`
- `rosbags`

### C++ Build Tooling

- standard ROS 2 C++ toolchain via `ament_cmake`

---

## Testing And Validation

The most useful repository-level validation flow is:

1. Build the workspace and confirm the relevant packages compile.
2. Launch the stack in simulation, on car, or against a recorded ROS bag.
3. Inspect `/scan`, `/drive_control`, and `/drive`.
4. Confirm that safety intervention occurs when forward clearance or TTC becomes unsafe.

### Bag Replay

```bash
ros2 launch milestones mpc_start_up.py
ros2 bag play <bag_path>
```

If replayed topics do not behave as expected, first verify:

- whether the bag publishes `/odom` or `/ego_racecar/odom`
- whether `sim` inside `milestones/launch/mpc_start_up.py` matches the intended mode
- whether the bag contains `/scan`

### Suggested Validation Cases

- straight corridor tracking
- corner entry and exit stability
- narrow-gap behavior
- obstacle approach with AEB intervention
- simulator versus on-car topic configuration

---

## Project Status

| Subsystem | Status | Summary |
|---|---|---|
| Local reactive MPC on car | **Achieved** | Stable closed-loop local control on the physical vehicle using LiDAR-driven geometry. |
| Global planning in simulation | **Achieved** | Track-level planning demonstrated on `sim_working_version`. |
| Global planning on car | **Partial** | Launches on hardware, but localization limits trajectory quality. |
| Experimental YOLO perception | **Experimental** | Standalone semantic perception exploration, not part of runtime stack. |

### Key Limitations

- **Hardware localization** remains the biggest blocker for robust on-car global planning.
- **No CUDA on vehicle** prevents real-time YOLO deployment.
- **Some metadata files** may still lag behind the active runtime stack and need cleanup for polished packaging.
