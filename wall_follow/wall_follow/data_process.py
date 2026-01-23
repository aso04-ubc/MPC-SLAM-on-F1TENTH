from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue

import numpy as np

from wall_follow.config import DEBUG
from wall_follow.KalmanFilter import SimpleKalmanFilter
from wall_follow.PID_control import PIDControl

class DataProcess(Node):

    PID_control : PIDControl

    def __init__(self):
        super().__init__("wall_follow")

        self.windows_size = 5

        
        # add kalman filter to smooth the lidar data
        # make the parameters tunable later
        self.simple_kalmanFilter = SimpleKalmanFilter(10.0, 0.1) 

        ## set up PID controller
        self.PID_control = PIDControl()

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

        if DEBUG:
            self.filtered_lidar_pub = self.create_publisher(
                LaserScan,
                "wall_follow/filtered_scan",
                10
            )

            self.debug_info_pub = self.create_publisher(
                DiagnosticStatus,
                "wall_follow/debug_info",
                10
            )


    def OnReceiveOdomInfo(self, odom_data : Odometry):
        # Process the odometry data
        pass


    def OnReceiveLaserInfo(self, lidar_data : LaserScan):
        
        range_np = self.simple_kalmanFilter.update(np.array(lidar_data.ranges))

        self.angle_min = lidar_data.angle_min
        self.angle_max = lidar_data.angle_max
        self.angle_increment = lidar_data.angle_increment

        self.process_lidar(range_np)

        if DEBUG:
            lidar_data.ranges = range_np.tolist()    
            self.filtered_lidar_pub.publish(lidar_data)

    def get_target_index(self,  target_angle : float):
        raw_index = (target_angle - self.angle_min) / self.angle_increment
        target_index = int(np.clip(raw_index, 0, (self.angle_max - self.angle_min) / self.angle_increment - 1))
        return target_index
        

    def process_lidar(self, lidar_data : np.ndarray):
        left_dist, left_tangent = self.process_lidar_one_side(
            lidar_data[self.get_target_index(0.0): self.get_target_index(np.pi/2)]
        )

        right_dist, right_tangent = self.process_lidar_one_side(
            lidar_data[self.get_target_index(-np.pi/2): self.get_target_index(0.0)]
        )

        if DEBUG:
            msg = DiagnosticStatus()
            msg.level = DiagnosticStatus.OK
            msg.name = "Wall Follow Debug Info"
            msg.values = [
                KeyValue(key="Left Distance", value=str(left_dist)),
                KeyValue(key="Left Tangent Angle", value=str(left_tangent)),
                KeyValue(key="Right Distance", value=str(right_dist)),
                KeyValue(key="Right Tangent Angle", value=str(right_tangent))
            ]
            self.debug_info_pub.publish(msg)


        
    def process_lidar_one_side(self, lidar_data : np.ndarray):

        valid_mask = (lidar_data > 0.01) & (lidar_data < 10.0)

        # Get the index of the closest point
        lidar_data[~valid_mask] = np.inf
        min_idx = np.argmin(lidar_data)

        start_idx = max(0, min_idx - self.windows_size)
        end_idx = min(len(lidar_data) - 1, min_idx + self.windows_size)

        local_ranges = lidar_data[start_idx:end_idx]
        local_angles = self.angle_min + np.arange(start_idx, end_idx) * self.angle_increment

        x = local_ranges * np.cos(local_angles)
        y = local_ranges * np.sin(local_angles)

        if len(x) < 3: 
            return local_ranges[min_idx - start_idx], local_angles[min_idx - start_idx]

        A_matrix = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A_matrix, y, rcond=None)[0]

        dist = abs(c) / np.sqrt(m**2 + 1)
        tangent_angle = np.arctan(m)

        return dist, tangent_angle

