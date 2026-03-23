from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from dev_b7_interfaces.msg import DriveControlMessage 

def wrap_drive_message(node, speed: float, steering: float, priority: int = 1004):
    """
    node: ROS2 node instance
    speed: Desired velocity in m/s
    steering: Desired steering angle in radians
    priority: Integer priority for the safety node
    """
    temp_msg = AckermannDriveStamped()
    
    temp_msg.header.stamp = node.get_clock().now().to_msg()
    temp_msg.header.frame_id = 'base_link'
    
    temp_msg.drive = AckermannDrive()
    temp_msg.drive.speed = float(speed)
    temp_msg.drive.steering_angle = float(steering)

    full_msg = DriveControlMessage()
    full_msg.active = True
    full_msg.priority = priority 
    full_msg.drive = temp_msg

    return full_msg