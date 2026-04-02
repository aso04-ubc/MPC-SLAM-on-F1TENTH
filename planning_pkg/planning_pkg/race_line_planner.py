#!/usr/bin/env python3
import math
import time
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from planning_pkg.race_line_core import (
    RaceLinePlan,
    occupancy_data_to_gray_image,
    plan_from_map,
    reindex_closed_raceline_by_pose,
    world_to_pixel,
)


class RaceLinePlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('race_line_planner')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('sim', True),
                ('replan_rate_hz', 1.5),
                ('map_topic', '/mapping/occupancy_grid'),
                ('pose_topic', '/mapping/fused_pose'),
                ('path_topic', '/race_line/global_path'),
                ('debug_marker_topic', '/race_line/debug_markers'),
                ('frame_id_default', 'map'),
                ('map_stale_timeout_s', 2.0),
                ('pose_stale_timeout_s', 2.0),
                ('pose_reindex_enabled', True),
                ('free_thresh', 200),
                ('occ_thresh', 90),
                ('sample_count', 320),
                ('w_curvature', 40.0),
                ('w_smooth', 8.0),
                ('w_center_bias', 1.0),
                ('v_max', 4.0),
                ('a_lat_max', 3.0),
                ('a_long_accel_max', 2.0),
                ('a_long_brake_max', 4.0),
            ],
        )

        self.sim = bool(self.get_parameter('sim').value)
        self.replan_rate_hz = float(self.get_parameter('replan_rate_hz').value)
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.debug_marker_topic = str(self.get_parameter('debug_marker_topic').value)
        self.frame_id_default = str(self.get_parameter('frame_id_default').value)
        self.map_stale_timeout_s = float(self.get_parameter('map_stale_timeout_s').value)
        self.pose_stale_timeout_s = float(self.get_parameter('pose_stale_timeout_s').value)
        self.pose_reindex_enabled = bool(self.get_parameter('pose_reindex_enabled').value)

        self.free_thresh = int(self.get_parameter('free_thresh').value)
        self.occ_thresh = int(self.get_parameter('occ_thresh').value)
        self.sample_count = int(self.get_parameter('sample_count').value)

        self.w_curvature = float(self.get_parameter('w_curvature').value)
        self.w_smooth = float(self.get_parameter('w_smooth').value)
        self.w_center_bias = float(self.get_parameter('w_center_bias').value)

        self.v_max = float(self.get_parameter('v_max').value)
        self.a_lat_max = float(self.get_parameter('a_lat_max').value)
        self.a_long_accel_max = float(self.get_parameter('a_long_accel_max').value)
        self.a_long_brake_max = float(self.get_parameter('a_long_brake_max').value)

        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.debug_marker_topic, 1)
        self.debug_image_pub = self.create_publisher(Image, '/race_line/debug_image', 1)

        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 1)
        self.pose_sub = self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)

        self.latest_map_gray: Optional[np.ndarray] = None
        self.latest_map_scale_px_per_m = 40.0
        self.latest_map_center_px_x = 500.0
        self.latest_map_center_px_y = 500.0
        self.latest_map_frame_id = self.frame_id_default
        self.latest_map_rx_time_s = 0.0

        self.latest_pose: Optional[PoseStamped] = None
        self.latest_pose_rx_time_s = 0.0

        self.last_plan: Optional[RaceLinePlan] = None
        self.last_good_path: Optional[Path] = None
        self.iteration = 0
        self.last_warn_no_map_s = 0.0
        self.last_warn_stale_map_s = 0.0
        self.last_warn_stale_pose_s = 0.0

        self.first_lap_complete = False
        self.start_pose_xy: Optional[tuple] = None
        self.last_pose_xy: Optional[tuple] = None
        self.total_distance_traveled = 0.0
        self.min_lap_distance_m = 5.0
        self.loop_closure_radius_m = 1.5

        self.window_name = 'Race Line Planner Debug'
        if self.sim:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 900)

        timer_period = 1.0 / max(0.1, self.replan_rate_hz)
        self.timer = self.create_timer(timer_period, self.replan_callback)

    def now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def map_callback(self, msg: OccupancyGrid) -> None:
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0:
            return

        if len(msg.data) != width * height:
            self.get_logger().warn(
                f'Ignoring map with invalid data length: {len(msg.data)} != {width * height}'
            )
            return

        self.latest_map_gray = occupancy_data_to_gray_image(
            occ_data=np.array(msg.data, dtype=np.int16),
            width=width,
            height=height,
            occupied_threshold=50,
        )

        resolution = float(msg.info.resolution)
        if resolution > 1e-6:
            self.latest_map_scale_px_per_m = 1.0 / resolution

        self.latest_map_center_px_x = 0.5 * width
        self.latest_map_center_px_y = 0.5 * height
        self.latest_map_frame_id = msg.header.frame_id if msg.header.frame_id else self.frame_id_default
        self.latest_map_rx_time_s = self.now_seconds()

    def pose_callback(self, msg: PoseStamped) -> None:
        self.latest_pose = msg
        self.latest_pose_rx_time_s = self.now_seconds()

        px = float(msg.pose.position.x)
        py = float(msg.pose.position.y)

        if self.start_pose_xy is None:
            self.start_pose_xy = (px, py)
            self.last_pose_xy = (px, py)
            return

        if self.last_pose_xy is not None:
            dx = px - self.last_pose_xy[0]
            dy = py - self.last_pose_xy[1]
            self.total_distance_traveled += math.hypot(dx, dy)
        self.last_pose_xy = (px, py)

        if not self.first_lap_complete:
            dist_to_start = math.hypot(px - self.start_pose_xy[0], py - self.start_pose_xy[1])
            if self.total_distance_traveled >= self.min_lap_distance_m and dist_to_start <= self.loop_closure_radius_m:
                self.first_lap_complete = True
                self.get_logger().info(
                    f'First lap complete (traveled {self.total_distance_traveled:.1f} m). '
                    'Now publishing reference road.'
                )

    def map_is_fresh(self, now_s: float) -> bool:
        if self.latest_map_gray is None:
            return False
        return (now_s - self.latest_map_rx_time_s) <= self.map_stale_timeout_s

    def pose_is_fresh(self, now_s: float) -> bool:
        if self.latest_pose is None:
            return False
        return (now_s - self.latest_pose_rx_time_s) <= self.pose_stale_timeout_s

    def replan_callback(self) -> None:
        now_s = self.now_seconds()

        if self.latest_map_gray is None:
            if now_s - self.last_warn_no_map_s > 1.5:
                self.get_logger().warn('Waiting for map on /mapping/occupancy_grid...')
                self.last_warn_no_map_s = now_s
            return

        if not self.map_is_fresh(now_s):
            if now_s - self.last_warn_stale_map_s > 1.5:
                self.get_logger().warn('Map is stale; reusing last good path if available.')
                self.last_warn_stale_map_s = now_s
            if self.first_lap_complete and self.last_good_path is not None:
                msg = self.last_good_path
                msg.header.stamp = self.get_clock().now().to_msg()
                self.path_pub.publish(msg)
            return

        t0 = time.perf_counter()
        try:
            plan = plan_from_map(
                gray=self.latest_map_gray,
                scale_px_per_m=self.latest_map_scale_px_per_m,
                map_center_px_x=self.latest_map_center_px_x,
                map_center_px_y=self.latest_map_center_px_y,
                free_thresh=self.free_thresh,
                occ_thresh=self.occ_thresh,
                w_curvature=self.w_curvature,
                w_smooth=self.w_smooth,
                w_center_bias=self.w_center_bias,
                v_max=self.v_max,
                a_lat_max=self.a_lat_max,
                a_long_accel_max=self.a_long_accel_max,
                a_long_brake_max=self.a_long_brake_max,
                sample_count=self.sample_count,
            )
        except Exception as exc:
            self.get_logger().error(f'Race line planning failed: {exc}')
            if self.first_lap_complete and self.last_good_path is not None:
                msg = self.last_good_path
                msg.header.stamp = self.get_clock().now().to_msg()
                self.path_pub.publish(msg)
            return

        if self.pose_reindex_enabled:
            if self.pose_is_fresh(now_s):
                pose = self.latest_pose.pose
                q = pose.orientation
                pose_yaw = np.arctan2(
                    2.0 * (q.w * q.z + q.x * q.y),
                    1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                )
                race_xy, race_yaw, race_speed = reindex_closed_raceline_by_pose(
                    raceline_xy=plan.raceline_xy,
                    yaw=plan.yaw,
                    speed_profile=plan.speed_profile,
                    pose_x=float(pose.position.x),
                    pose_y=float(pose.position.y),
                    pose_yaw=float(pose_yaw),
                    enforce_heading_alignment=True,
                )
                plan.raceline_xy = race_xy
                plan.yaw = race_yaw
                plan.speed_profile = race_speed
                plan.raceline_px = world_to_pixel(
                    race_xy,
                    self.latest_map_scale_px_per_m,
                    self.latest_map_center_px_x,
                    self.latest_map_center_px_y,
                )
            elif now_s - self.last_warn_stale_pose_s > 1.5:
                self.get_logger().warn('Pose is stale; publishing non-reindexed race line.')
                self.last_warn_stale_pose_s = now_s

        self.last_plan = plan

        if self.first_lap_complete:
            path_msg = self.publish_path(plan)
            self.last_good_path = path_msg

        self.draw_debug(plan)
        if not self.sim:
            self.publish_markers(plan)

        self.iteration += 1
        if self.iteration % 10 == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.get_logger().info(
                f'Published race line with {len(plan.raceline_xy)} points in {dt_ms:.1f} ms '
                f'(v=[{float(np.min(plan.speed_profile)):.2f}, {float(np.max(plan.speed_profile)):.2f}] m/s)'
            )

    def publish_path(self, plan: RaceLinePlan) -> Path:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.latest_map_frame_id

        for (x, y), yaw in zip(plan.raceline_xy, plan.yaw):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = float(np.sin(0.5 * yaw))
            pose.pose.orientation.w = float(np.cos(0.5 * yaw))
            path.poses.append(pose)

        self.path_pub.publish(path)
        return path

    def publish_markers(self, plan: RaceLinePlan) -> None:
        stamp = self.get_clock().now().to_msg()

        center = Marker()
        center.header.frame_id = self.latest_map_frame_id
        center.header.stamp = stamp
        center.ns = 'race_line'
        center.id = 0
        center.type = Marker.LINE_STRIP
        center.action = Marker.ADD
        center.scale.x = 0.04
        center.color.a = 1.0
        center.color.r = 0.1
        center.color.g = 0.6
        center.color.b = 1.0
        center.pose.orientation.w = 1.0

        for x, y in plan.centerline_xy:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.03
            center.points.append(p)
        if plan.centerline_xy.shape[0] > 0:
            p = Point()
            p.x = float(plan.centerline_xy[0, 0])
            p.y = float(plan.centerline_xy[0, 1])
            p.z = 0.03
            center.points.append(p)

        race = Marker()
        race.header.frame_id = self.latest_map_frame_id
        race.header.stamp = stamp
        race.ns = 'race_line'
        race.id = 1
        race.type = Marker.LINE_STRIP
        race.action = Marker.ADD
        race.scale.x = 0.06
        race.color.a = 1.0
        race.color.r = 1.0
        race.color.g = 0.1
        race.color.b = 0.8
        race.pose.orientation.w = 1.0

        for x, y in plan.raceline_xy:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            race.points.append(p)
        if plan.raceline_xy.shape[0] > 0:
            p = Point()
            p.x = float(plan.raceline_xy[0, 0])
            p.y = float(plan.raceline_xy[0, 1])
            p.z = 0.05
            race.points.append(p)

        arr = MarkerArray()
        arr.markers = [center, race]
        self.marker_pub.publish(arr)

    def draw_debug(self, plan: RaceLinePlan) -> None:
        canvas = cv2.cvtColor(self.latest_map_gray, cv2.COLOR_GRAY2BGR)

        cv2.polylines(
            canvas,
            [np.round(plan.centerline_px).astype(np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=(255, 180, 0),
            thickness=2,
        )

        cv2.polylines(
            canvas,
            [np.round(plan.raceline_px).astype(np.int32).reshape(-1, 1, 2)],
            isClosed=True,
            color=(255, 0, 255),
            thickness=2,
        )

        if self.latest_pose is not None:
            pose = self.latest_pose.pose.position
            px = int(round(self.latest_map_center_px_x + pose.y * self.latest_map_scale_px_per_m))
            py = int(round(self.latest_map_center_px_y - pose.x * self.latest_map_scale_px_per_m))
            if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)

        text = (
            f'pts={len(plan.raceline_xy)} '
            f'v_min={float(np.min(plan.speed_profile)):.2f} '
            f'v_max={float(np.max(plan.speed_profile)):.2f}'
        )
        cv2.putText(canvas, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 255), 2)

        if self.sim:
            cv2.imshow(self.window_name, canvas)
            cv2.waitKey(1)
        else:
            self.debug_image_pub.publish(self._numpy_to_image_msg(canvas))

    def _numpy_to_image_msg(self, img: np.ndarray) -> Image:
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = img.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = img.shape[1] * 3
        msg.data = img.tobytes()
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RaceLinePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.sim:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
