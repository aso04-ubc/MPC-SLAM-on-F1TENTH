from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

import rclpy
from rclpy.node import Node

class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")
        self.subscriptions = self.create_subscription(
            LaserScan,
            "scan",
            self.OnReceiveLaserInfo,
            10
        )

    def OnReceiveLaserInfo(self, info : LaserScan):
        self.get_logger().info(info)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = SafetyNode()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()