# Safety Python Node

Python implementation of the safety node with Autonomous Emergency Braking (AEB) and priority-based command arbitration.

## Features

### 1. Autonomous Emergency Braking (AEB)
- Monitors laser scan data for obstacles
- Calculates Time-To-Collision (TTC) for each scan point
- Engages emergency braking when:
  - TTC < threshold (default: 0.5s), OR
  - Distance < threshold (default: 0.25m)
- Publishes stop commands at 200 Hz when active
- Preserves steering angle from last command

### 2. Priority-Based Command Arbitration
- Accepts drive commands from multiple sources
- Each command has a priority level
- Higher priority commands override lower ones
- Commands can be activated/deactivated dynamically
- Only the highest priority active command is published

## Subscribed Topics

- `/drive_control` (dev_b7_interfaces/DriveControlMessage): Drive commands with priority
- `/scan` (sensor_msgs/LaserScan): Lidar data for obstacle detection
- `/ego_racecar/odom` (nav_msgs/Odometry): Vehicle odometry for speed

## Published Topics

- `/drive` (ackermann_msgs/AckermannDriveStamped): Final drive commands to vehicle

## Parameters

- `ttc_threshold` (float, default: 0.5): Time-to-collision threshold in seconds
- `distance_threshold` (float, default: 0.25): Minimum distance threshold in meters

## Usage

### Build the package
```bash
cd ~/sim_ws
colcon build --packages-select safety_python
source install/setup.bash
```

### Run with default parameters
```bash
ros2 run safety_python safety_python_node
```

### Run with custom parameters
```bash
ros2 run safety_python safety_python_node --ros-args \
  -p ttc_threshold:=0.6 \
  -p distance_threshold:=0.3
```

## Algorithm Details

### Time-To-Collision (TTC) Calculation

For each laser scan point:
1. Calculate angle: `angle = angle_min + i * angle_increment`
2. Calculate range rate: `range_rate = vehicle_speed * cos(angle)`
3. Calculate TTC: `ttc = distance / range_rate` (if approaching)
4. Track minimum TTC across all points

### Priority Arbitration

- Commands stored in a dictionary: `{priority: DriveControlMessage}`
- Active commands are added/updated
- Inactive commands are removed
- Highest priority (max key) is published
- AEB overrides all commands when active

## Differences from C++ Version

This Python implementation provides the same core safety logic with these simplifications:

- **No multithreading**: Uses ROS2 timers instead of separate threads
- **No lifecycle management**: Standard ROS2 node (not lifecycle node)
- **No console input**: Removed stdin listening and manual AEB control
- **No argument parsing**: Uses ROS2 parameters instead
- **Simpler state management**: No complex atomic operations or mutexes needed

The core AEB and arbitration logic remains identical to the C++ version.
