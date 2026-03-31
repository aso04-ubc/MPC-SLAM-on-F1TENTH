#!/usr/bin/env python3
"""YOLO-based high-level planner for dynamic gap-following tuning."""

from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - runtime dependency
    YOLO = None

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
except Exception:  # pragma: no cover - optional during non-ROS linting
    PackageNotFoundError = Exception
    get_package_share_directory = None


class YoloHighLevelPlanner(Node):
    """Run object detection and publish dynamic gap-following limits."""

    def __init__(self):
        super().__init__('yolo_high_level_planner')

        self.declare_parameter('sim', False)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'overlay_topic',
            '/high_level_planner/yolo_overlay',
        )
        self.declare_parameter(
            'gap_following_param_topic',
            '/high_level_planner/gap_following_params',
        )
        self.declare_parameter(
            'detection_count_topic',
            '/high_level_planner/detection_counts',
        )
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('log_frequency', 20)
        self.declare_parameter('obstacle_class_name', 'Obstacle')

        # Tuning targets sent to gap_following:
        # [max_speed, distance_slowdown_threshold,
        #  obstacle_flag, obstacle_count]
        self.declare_parameter('clear_max_speed', 3.0)
        self.declare_parameter('obstacle_max_speed', 1.2)
        self.declare_parameter('clear_distance_slowdown_threshold', 1.6)
        self.declare_parameter('obstacle_distance_slowdown_threshold', 2.6)

        self.sim = bool(self.get_parameter('sim').value)
        self.image_topic = str(self.get_parameter('image_topic').value)
        self.overlay_topic = str(self.get_parameter('overlay_topic').value)
        self.gap_following_param_topic = str(
            self.get_parameter('gap_following_param_topic').value
        )
        self.detection_count_topic = str(
            self.get_parameter('detection_count_topic').value
        )
        configured_model_path = str(self.get_parameter('model_path').value)
        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.log_frequency = max(
            1,
            int(self.get_parameter('log_frequency').value),
        )
        self.obstacle_class_name = str(
            self.get_parameter('obstacle_class_name').value
        ).lower()

        self.clear_max_speed = max(
            0.0,
            float(self.get_parameter('clear_max_speed').value),
        )
        self.obstacle_max_speed = max(
            0.0,
            float(self.get_parameter('obstacle_max_speed').value),
        )
        self.clear_distance_slowdown_threshold = max(
            0.01,
            float(
                self.get_parameter(
                    'clear_distance_slowdown_threshold'
                ).value
            ),
        )
        self.obstacle_distance_slowdown_threshold = max(
            0.01,
            float(
                self.get_parameter(
                    'obstacle_distance_slowdown_threshold'
                ).value
            ),
        )

        if YOLO is None:
            raise RuntimeError(
                'ultralytics is not installed. '
                'Install it with: pip install ultralytics'
            )

        model_path = self._resolve_model_path(configured_model_path)
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.model = YOLO(model_path)

        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.tuning_pub = self.create_publisher(
            Float32MultiArray,
            self.gap_following_param_topic,
            10,
        )
        self.count_pub = self.create_publisher(
            String,
            self.detection_count_topic,
            10,
        )

        self.overlay_pub = None
        if not self.sim:
            self.overlay_pub = self.create_publisher(
                Image,
                self.overlay_topic,
                10,
            )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_image,
        )

        self._frame_count = 0
        self._last_obstacle_present = None
        self._latest_max_speed = self.clear_max_speed
        self._latest_distance_slowdown = self.clear_distance_slowdown_threshold
        self._latest_obstacle_count = 0

        # Republish at low rate so new subscribers get current tuning.
        self.create_timer(1.0, self._republish_latest_tuning)

        self.get_logger().info(
            'YOLO high-level planner started '
            f'(sim={self.sim}, image_topic={self.image_topic}, '
            f'tune_topic={self.gap_following_param_topic})'
        )

    def _resolve_model_path(self, configured_model_path):
        """Resolve model path from parameter, install, or source tree."""
        if configured_model_path:
            model_path = Path(configured_model_path).expanduser()
            if model_path.exists():
                return str(model_path)
            self.get_logger().warning(
                f'Model path parameter does not exist: {model_path}. '
                'Falling back to defaults.'
            )

        if get_package_share_directory is not None:
            try:
                installed_model = (
                    Path(get_package_share_directory('high_level_planner'))
                    / 'resource'
                    / 'best_with_road.pt'
                )
                if installed_model.exists():
                    return str(installed_model)
            except PackageNotFoundError:
                pass

        source_model = (
            Path(__file__).resolve().parents[1]
            / 'resource'
            / 'best_with_road.pt'
        )
        if source_model.exists():
            return str(source_model)

        raise FileNotFoundError(
            'Cannot find YOLO model. '
            'Set model_path parameter or place best_with_road.pt in resource/.'
        )

    def _encoding_metadata(self, encoding):
        enc = encoding.lower()
        if enc == 'bgr8':
            return np.uint8, 3, None
        if enc == 'rgb8':
            return np.uint8, 3, 'rgb2bgr'
        if enc == 'mono8':
            return np.uint8, 1, 'mono2bgr'
        raise ValueError(f'Unsupported encoding: {encoding}')

    def _ros_image_to_cv2(self, msg):
        height, width, step = msg.height, msg.width, msg.step
        dtype, channels, conversion = self._encoding_metadata(msg.encoding)

        if height == 0 or width == 0 or step == 0:
            raise ValueError('Image has zero dimension')

        expected_stride = width * channels * np.dtype(dtype).itemsize
        if step != expected_stride:
            raise ValueError(
                'Unexpected row stride: '
                f'expected {expected_stride}, got {step}'
            )

        flat = np.frombuffer(msg.data, dtype=np.uint8, count=height * step)
        if channels == 1:
            image = flat.view(dtype).reshape((height, width))
        else:
            image = flat.view(dtype).reshape((height, width, channels))

        if conversion == 'rgb2bgr':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif conversion == 'mono2bgr':
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image

    def _cv2_to_ros_image(self, cv_image, header):
        msg = Image()
        msg.header = header
        msg.height = cv_image.shape[0]
        msg.width = cv_image.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = cv_image.shape[1] * 3
        msg.data = cv_image.tobytes()
        return msg

    def _class_name(self, class_id):
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(class_id, f'class_{class_id}'))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return f'class_{class_id}'

    def _extract_detection_counts(self, results):
        counts = Counter()
        if results.boxes is None or len(results.boxes) == 0:
            return counts

        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        for cls_id in cls_ids:
            counts[self._class_name(int(cls_id))] += 1
        return counts

    def _draw_overlay(
        self,
        image,
        results,
        max_speed,
        distance_slowdown_threshold,
    ):
        overlay = image.copy()
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1] - 1, x2)
                y2 = min(image.shape[0] - 1, y2)

                class_name = self._class_name(int(cls_id))
                if class_name.lower() == self.obstacle_class_name:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                label = f'{class_name} {conf:.2f}'

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    overlay,
                    label,
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        status = (
            f'max_speed={max_speed:.2f} '
            f'distance_slowdown_threshold={distance_slowdown_threshold:.2f}'
        )
        cv2.putText(
            overlay,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return overlay

    def _publish_counts(self, counts, obstacle_count):
        payload = {
            'total': int(sum(counts.values())),
            'obstacle': int(obstacle_count),
            'counts': dict(sorted(counts.items())),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.count_pub.publish(msg)

    def _publish_tuning(
        self,
        max_speed,
        distance_slowdown_threshold,
        obstacle_count,
    ):
        tune_msg = Float32MultiArray()
        tune_msg.data = [
            float(max_speed),
            float(distance_slowdown_threshold),
            1.0 if obstacle_count > 0 else 0.0,
            float(obstacle_count),
        ]
        self.tuning_pub.publish(tune_msg)

    def _republish_latest_tuning(self):
        self._publish_tuning(
            self._latest_max_speed,
            self._latest_distance_slowdown,
            self._latest_obstacle_count,
        )

    def image_callback(self, image_msg):
        try:
            cv_image = self._ros_image_to_cv2(image_msg)
        except Exception as exc:
            self.get_logger().error(f'Image conversion failed: {exc}')
            return

        try:
            result = self.model.predict(
                source=cv_image,
                conf=self.confidence_threshold,
                verbose=False,
            )[0]
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        counts = self._extract_detection_counts(result)
        obstacle_count = 0
        for class_name, class_count in counts.items():
            if class_name.lower() == self.obstacle_class_name:
                obstacle_count += int(class_count)

        obstacle_present = obstacle_count > 0
        if obstacle_present:
            max_speed = self.obstacle_max_speed
            distance_slowdown_threshold = (
                self.obstacle_distance_slowdown_threshold
            )
        else:
            max_speed = self.clear_max_speed
            distance_slowdown_threshold = (
                self.clear_distance_slowdown_threshold
            )

        self._latest_max_speed = max_speed
        self._latest_distance_slowdown = distance_slowdown_threshold
        self._latest_obstacle_count = obstacle_count

        self._publish_tuning(
            max_speed,
            distance_slowdown_threshold,
            obstacle_count,
        )
        self._publish_counts(counts, obstacle_count)

        self._frame_count += 1
        should_log = (
            self._frame_count % self.log_frequency == 0
            or self._last_obstacle_present is None
            or obstacle_present != self._last_obstacle_present
        )
        if should_log:
            if counts:
                count_str = ', '.join(
                    f'{k}:{v}'
                    for k, v in sorted(counts.items())
                )
            else:
                count_str = 'none:0'
            self.get_logger().info(
                f'Detected counts -> {count_str}; obstacle={obstacle_count}; '
                f'max_speed={max_speed:.2f}; '
                'distance_slowdown_threshold='
                f'{distance_slowdown_threshold:.2f}'
            )

        self._last_obstacle_present = obstacle_present

        overlay = self._draw_overlay(
            cv_image,
            result,
            max_speed,
            distance_slowdown_threshold,
        )

        if self.sim:
            cv2.imshow('YOLO High Level Planner', overlay)
            cv2.waitKey(1)
        elif self.overlay_pub is not None:
            self.overlay_pub.publish(
                self._cv2_to_ros_image(overlay, image_msg.header)
            )

    def destroy_node(self):
        if self.sim:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloHighLevelPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
