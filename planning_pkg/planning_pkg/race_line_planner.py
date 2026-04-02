#!/usr/bin/env python3
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
                # Keep the track's start index fixed (do not roll by current pose),
                # so the vehicle "returns" to the same start point after completing the loop.
                ('pose_reindex_enabled', False),
                # Use the extracted track centerline directly as the reference line.
                ('use_centerline_as_raceline', True),
                ('centerline_smooth_passes', 4),
                # When sim=False, publish debug images to a topic.
                ('publish_debug_images', True),
                ('race_line_debug_image_topic', '/race_line/debug_image'),
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
                ('require_closed_track', True),
                ('closed_track_min_inner_area_px', 800.0),
                ('use_precomputed_path', False),
                ('precomputed_path_file', ''),
                ('auto_save_precomputed_path', False),
                ('auto_save_filepath', ''),
            ],
        )

        self.sim = bool(self.get_parameter('sim').value)
        self.use_precomputed_path = bool(self.get_parameter('use_precomputed_path').value)
        self.precomputed_path_file = str(self.get_parameter('precomputed_path_file').value)
        self.auto_save_precomputed_path = bool(self.get_parameter('auto_save_precomputed_path').value)
        self.auto_save_filepath = str(self.get_parameter('auto_save_filepath').value)
        self.replan_rate_hz = float(self.get_parameter('replan_rate_hz').value)
        self.map_topic = str(self.get_parameter('map_topic').value)
        self.pose_topic = str(self.get_parameter('pose_topic').value)
        self.path_topic = str(self.get_parameter('path_topic').value)
        self.debug_marker_topic = str(self.get_parameter('debug_marker_topic').value)
        self.frame_id_default = str(self.get_parameter('frame_id_default').value)
        self.map_stale_timeout_s = float(self.get_parameter('map_stale_timeout_s').value)
        self.pose_stale_timeout_s = float(self.get_parameter('pose_stale_timeout_s').value)
        self.pose_reindex_enabled = bool(self.get_parameter('pose_reindex_enabled').value)
        self.use_centerline_as_raceline = bool(self.get_parameter('use_centerline_as_raceline').value)
        self.centerline_smooth_passes = int(self.get_parameter('centerline_smooth_passes').value)
        self.publish_race_line_debug_images = (
            (not self.sim) and bool(self.get_parameter('publish_debug_images').value)
        )
        self.race_line_debug_image_topic = str(self.get_parameter('race_line_debug_image_topic').value)
        self.race_line_debug_image_pub = None
        if self.publish_race_line_debug_images:
            self.race_line_debug_image_pub = self.create_publisher(Image, self.race_line_debug_image_topic, 1)

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
        self.require_closed_track = bool(self.get_parameter('require_closed_track').value)
        self.closed_track_min_inner_area_px = float(self.get_parameter('closed_track_min_inner_area_px').value)
        # Once we have confirmed the loop is closed at least once, start publishing
        # race line guidance continuously to MPC. This avoids MPC "stalling"
        # when the closure metric briefly drops due to map noise.
        self.loop_closed_observed = False

        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.marker_pub = self.create_publisher(MarkerArray, self.debug_marker_topic, 1)

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
        self.last_warn_open_track_s = 0.0

        # Precomputed path data
        self.precomputed_xy: Optional[np.ndarray] = None
        self.precomputed_yaw: Optional[np.ndarray] = None
        self.precomputed_speeds: Optional[np.ndarray] = None
        self.precomputed_frame_id: str = self.frame_id_default

        # Load precomputed path if enabled
        if self.use_precomputed_path:
            if not self.precomputed_path_file:
                self.get_logger().error('use_precomputed_path=True but precomputed_path_file is empty')
            else:
                self.load_precomputed_path()

        self.window_name = 'Race Line Planner Debug'
        if self.sim:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 900)

        timer_period = 1.0 / max(0.1, self.replan_rate_hz)
        self.timer = self.create_timer(timer_period, self.replan_callback)

        mode_str = 'precomputed path' if self.use_precomputed_path else 'live SLAM planning'
        if not self.use_precomputed_path and self.auto_save_precomputed_path:
            mode_str += ' (auto-saving)'
        self.get_logger().info(f'Race line planner initialized in {mode_str} mode')

    def now_seconds(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def load_precomputed_path(self) -> None:
        """Load precomputed path from JSON file."""
        import json

        try:
            with open(self.precomputed_path_file, 'r') as f:
                data = json.load(f)

            self.precomputed_xy = np.array(data['raceline_xy'], dtype=float)
            self.precomputed_yaw = np.array(data['yaw'], dtype=float)
            self.precomputed_speeds = np.array(
                data.get('speed_profile', np.ones(len(self.precomputed_xy))), dtype=float
            )
            self.precomputed_frame_id = data.get('frame_id', self.frame_id_default)

            self.get_logger().info(
                f'Loaded precomputed path from {self.precomputed_path_file}: '
                f'{len(self.precomputed_xy)} waypoints'
            )
        except FileNotFoundError:
            self.get_logger().error(f'Precomputed path file not found: {self.precomputed_path_file}')
            self.use_precomputed_path = False
        except (json.JSONDecodeError, KeyError) as e:
            self.get_logger().error(f'Error parsing precomputed path file: {e}')
            self.use_precomputed_path = False

    def save_plan_as_precomputed(self, plan: RaceLinePlan, filepath: str) -> None:
        """Save a RaceLinePlan as a precomputed path JSON file."""
        import json

        try:
            precomputed_data = {
                'raceline_xy': plan.raceline_xy.tolist(),
                'yaw': plan.yaw.tolist(),
                'speed_profile': plan.speed_profile.tolist(),
                'frame_id': self.latest_map_frame_id,
            }

            with open(filepath, 'w') as f:
                json.dump(precomputed_data, f, indent=2)

            self.get_logger().info(
                f'Saved precomputed path to {filepath} with {len(plan.raceline_xy)} waypoints'
            )

            # Also save CSV for easy debugging/visualization
            csv_filepath = filepath.replace('.json', '.csv')
            self.save_plan_as_csv(plan, csv_filepath)

            # Also save debug image if in simulation mode
            if self.sim:
                img_filepath = filepath.replace('.json', '_debug.png')
                self.save_debug_image(plan, img_filepath)

        except Exception as e:
            self.get_logger().error(f'Failed to save precomputed path: {e}')

    def save_plan_as_csv(self, plan: RaceLinePlan, filepath: str) -> None:
        """Save a RaceLinePlan as a CSV file for easy debugging/visualization."""
        try:
            import csv

            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'x', 'y', 'yaw', 'speed'])

                for i, ((x, y), yaw, speed) in enumerate(zip(plan.raceline_xy, plan.yaw, plan.speed_profile)):
                    writer.writerow([i, f'{x:.3f}', f'{y:.3f}', f'{yaw:.3f}', f'{speed:.3f}'])

            self.get_logger().info(f'Saved debug CSV to {filepath}')

        except Exception as e:
            self.get_logger().error(f'Failed to save debug CSV: {e}')

    def save_debug_image(self, plan: RaceLinePlan, filepath: str) -> None:
        """Save a debug visualization image of the planned path."""
        try:
            # Create the same debug canvas as draw_debug
            canvas = cv2.cvtColor(self.latest_map_gray, cv2.COLOR_GRAY2BGR)

            # Draw centerline (orange)
            cv2.polylines(
                canvas,
                [np.round(plan.centerline_px).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True,
                color=(255, 180, 0),
                thickness=2,
            )

            # Draw racing line (magenta)
            cv2.polylines(
                canvas,
                [np.round(plan.raceline_px).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True,
                color=(255, 0, 255),
                thickness=2,
            )

            # Draw current pose if available (cyan)
            if self.latest_pose is not None:
                pose = self.latest_pose.pose.position
                px = int(round(self.latest_map_center_px_x + pose.y * self.latest_map_scale_px_per_m))
                py = int(round(self.latest_map_center_px_y - pose.x * self.latest_map_scale_px_per_m))
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)

            # Add text info
            text = (
                f'pts={len(plan.raceline_xy)} '
                f'v_min={float(np.min(plan.speed_profile)):.2f} '
                f'v_max={float(np.max(plan.speed_profile)):.2f}'
            )
            cv2.putText(canvas, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 255), 2)

            # Save the image
            cv2.imwrite(filepath, canvas)
            self.get_logger().info(f'Saved debug image to {filepath}')

        except Exception as e:
            self.get_logger().error(f'Failed to save debug image: {e}')

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

    def map_is_fresh(self, now_s: float) -> bool:
        if self.latest_map_gray is None:
            return False
        return (now_s - self.latest_map_rx_time_s) <= self.map_stale_timeout_s

    def pose_is_fresh(self, now_s: float) -> bool:
        if self.latest_pose is None:
            return False
        return (now_s - self.latest_pose_rx_time_s) <= self.pose_stale_timeout_s

    def track_is_closed(self, plan: RaceLinePlan) -> bool:
        if not self.require_closed_track:
            return True

        contours, hierarchy = cv2.findContours(plan.drivable_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours or hierarchy is None:
            return False

        h = hierarchy[0]
        outer_idxs = [i for i in range(len(contours)) if h[i, 3] == -1]
        if not outer_idxs:
            return False

        outer_idx = max(outer_idxs, key=lambda i: cv2.contourArea(contours[i]))
        child_idxs = [i for i in range(len(contours)) if h[i, 3] == outer_idx]
        if not child_idxs:
            return False

        max_inner_area = max(float(cv2.contourArea(contours[i])) for i in child_idxs)
        return max_inner_area >= self.closed_track_min_inner_area_px

    def replan_callback(self) -> None:
        now_s = self.now_seconds()

        # Handle precomputed path mode
        if self.use_precomputed_path:
            if self.precomputed_xy is None:
                self.get_logger().warn('Waiting for precomputed path to load...')
                return
            self.publish_precomputed_path()
            return

        # Handle live SLAM planning mode
        if self.latest_map_gray is None:
            if now_s - self.last_warn_no_map_s > 1.5:
                self.get_logger().warn('Waiting for map on /mapping/occupancy_grid...')
                self.last_warn_no_map_s = now_s
            return

        if not self.map_is_fresh(now_s):
            if now_s - self.last_warn_stale_map_s > 1.5:
                self.get_logger().warn('Map is stale; reusing last good path if available.')
                self.last_warn_stale_map_s = now_s
            if self.last_good_path is not None:
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
                use_centerline_raceline=self.use_centerline_as_raceline,
                centerline_smooth_passes=self.centerline_smooth_passes,
            )
        except Exception as exc:
            self.get_logger().error(f'Race line planning failed: {exc}')
            if self.last_good_path is not None:
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

        if self.require_closed_track and not self.loop_closed_observed:
            if not self.track_is_closed(plan):
                if now_s - self.last_warn_open_track_s > 1.5:
                    self.get_logger().warn(
                        'Track is not closed yet; withholding race line publish until loop closure.'
                    )
                    self.last_warn_open_track_s = now_s
                if self.sim or self.publish_race_line_debug_images:
                    self.draw_debug(plan)
                return
            # First time we confirm closure: start publishing to MPC.
            self.loop_closed_observed = True

        self.last_plan = plan
        path_msg = self.publish_path(plan)
        self.last_good_path = path_msg

        # Auto-save precomputed path if enabled
        if self.auto_save_precomputed_path and self.auto_save_filepath:
            self.save_plan_as_precomputed(plan, self.auto_save_filepath)

        if self.sim:
            self.draw_debug(plan)
        else:
            self.publish_markers(plan)
            if self.publish_race_line_debug_images:
                self.draw_debug(plan)

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

    def publish_precomputed_path(self) -> None:
        """Publish the precomputed path as a ROS Path message."""
        if self.precomputed_xy is None or self.precomputed_yaw is None:
            return
        
        now_s = self.now_seconds()
        race_pos = self.precomputed_xy.copy()
        race_yaw = self.precomputed_yaw.copy()

        if self.pose_reindex_enabled and self.pose_is_fresh(now_s):
            if self.latest_pose is not None:
                pose = self.latest_pose.pose
                q = pose.orientation
                pose_yaw = np.arctan2(2.0 * (q.w*q.z + q.x+q.y), 1.0 - 2.0 *(q.y*q.y + q.z*q.z))

                race_pos, race_yaw, _ = reindex_closed_raceline_by_pose(
                    raceline_xy=self.precomputed_xy,
                    yaw=self.precomputed_yaw,
                    speed_profile=self.precomputed_speeds,
                    pose_x=float(pose.position.x),
                    pose_y=float(pose.position.y),
                    pose_yaw=float(pose_yaw),
                    enforce_heading_alignment=True
                )

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.precomputed_frame_id

        for (x, y), yaw in zip(self.precomputed_xy, self.precomputed_yaw):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = float(np.sin(0.5 * yaw))
            pose.pose.orientation.w = float(np.cos(0.5 * yaw))
            path.poses.append(pose)

        self.path_pub.publish(path)
        self.last_good_path = path

        self.iteration += 1
        if self.iteration % 10 == 0:
            self.get_logger().info(
                f'Published precomputed race line with {len(self.precomputed_xy)} points '
                f'(from {self.precomputed_path_file})'
            )

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
        if self.publish_race_line_debug_images and self.race_line_debug_image_pub is not None:
            self.publish_debug_image(canvas)

    def publish_debug_image(self, canvas_bgr: np.ndarray) -> None:
        if self.race_line_debug_image_pub is None:
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.height = int(canvas_bgr.shape[0])
        msg.width = int(canvas_bgr.shape[1])
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = int(canvas_bgr.shape[1] * canvas_bgr.shape[2])
        msg.data = canvas_bgr.tobytes()
        self.race_line_debug_image_pub.publish(msg)


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
