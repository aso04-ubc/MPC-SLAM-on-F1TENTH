from rclpy.node import Node
from sim_ws.src.wall_follow.wall_follow.PID_control import PIDControl
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class DataProcess(Node):

    PID_control : PIDControl

    def __init__(self):
        super().__init__("data_process_node")

        # make sure to get most recent messages
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/ego_racecar/odom",
            self.OnReceiveOdomInfo,
            sensor_qos
        )
        
        self.laser_receiver = self.create_subscription(
            LaserScan,
            "scan",
            self.OnReceiveLaserInfo,
            sensor_qos
        )



        # TBD 
        # self.control_info_pusher = self.create_publisher(
        #     AckermannDriveStamped,
        #     'drive',
        #     10
        # )

        self.PID_control = PIDControl()

        raise NotImplementedError("This is a placeholder for the data processing implementation.")

    def OnReceiveOdomInfo(self, odom_data : Odometry):
        # Process the odometry data
        pass

    def OnReceiveLaserInfo(self, lidar_data : LaserScan):
        pass

    def process_lidar(self, lidar_data):
        # Process the lidar data
        pass

