# dev_b7_interfaces

This package defines custom ROS 2 message interfaces for the F1TENTH autonomous racing pipeline. It is specifically designed to support a multiplexing (mux) system that arbitrates between multiple control sources based on priority.

## 1. Message Definition

### ControlSubmissionMessage.msg

This message wraps the standard Ackermann drive command with a priority integer. This allows a central mux node to decide which controller (e.g., Safety, Wall Follower, or Planner) should currently command the car.

**Field Definitions:**
- `ackermann_msgs/AckermannDriveStamped drive`: The standard steering and speed command.
- `int32 priority`: A numerical value where higher integers indicate higher precedence.

---

## 2. Prerequisites

The package depends on the standard `ackermann_msgs` package. Ensure it is installed:

```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-ackermann-msgs
```

---

## 3. Build Instructions

To compile the interfaces within your ROS 2 workspace:

1. **Navigate to the workspace root:**
   ```bash
   cd ~/ros2_ws
   ```

2. **Build the specific package:**
   ```bash
   colcon build --packages-select dev_b7_interfaces --symlink-install
   ```

3. **Source the setup script:**
   ```bash
   source install/setup.bash
   ```

4. **Verify the message is recognized:**
   ```bash
   ros2 interface show dev_b7_interfaces/msg/ControlSubmissionMessage
   ```

---

## 4. Python Usage Example

The following snippet demonstrates how to import and publish this custom message in a ROS 2 Python node.

```python
import rclpy
from rclpy.node import Node
from dev_b7_interfaces.msg import ControlSubmissionMessage
from ackermann_msgs.msg import AckermannDriveStamped

class ControlClient(Node):
    def __init__(self):
        super().__init__('control_client_node')
        self.publisher = self.create_publisher(
            ControlSubmissionMessage, 
            '/mux/input', 
            10
        )
        self.timer = self.create_timer(0.1, self.publish_cmd)

    def publish_cmd(self):
        # 1. Create the container message
        msg = ControlSubmissionMessage()
        
        # 2. Populate the Ackermann drive command
        drive_data = AckermannDriveStamped()
        drive_data.header.stamp = self.get_clock().now().to_msg()
        drive_data.header.frame_id = "base_link"
        drive_data.drive.speed = 1.0
        drive_data.drive.steering_angle = 0.0
        
        # 3. Set the priority and bundle the command
        msg.drive = drive_data
        msg.priority = 50  # Example priority level
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published with priority {msg.priority}')

def main():
    rclpy.init()
    node = ControlClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 5. Development Reference

### package.xml
Ensure these lines are present to handle the interface generation and dependencies:

```xml
<depend>ackermann_msgs</depend>
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

### CMakeLists.txt
Add these blocks to your CMake configuration:

```cmake
find_package(rosidl_default_generators REQUIRED)
find_package(ackermann_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControlSubmissionMessage.msg"
  DEPENDENCIES ackermann_msgs
)
```