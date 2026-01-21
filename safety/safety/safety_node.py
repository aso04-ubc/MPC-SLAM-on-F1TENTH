from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
import rclpy

g_MinimumAcceptableDistance = 0.3

# 
class SafetyNode(Node):
    def __init__(self):
        super().__init__("safety_node")

        self.scan_subscription = self.create_subscription(  # Subscribe to the scan topic
                LaserScan, "/scan", 
                self.OnProcessLaserScanInfo, 10)
        

        self.controll_publisher = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)

    def OnProcessLaserScanInfo(self, info: LaserScan):
        minDistance = min(info.ranges);
        print(f"Min distance:{minDistance}")

        if (minDistance < g_MinimumAcceptableDistance):
            msg = AckermannDriveStamped()
            msg.drive.speed = 0.0
            print("Stop Requested!")
            self.controll_publisher.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)

    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    print("Safety node running")
    main()