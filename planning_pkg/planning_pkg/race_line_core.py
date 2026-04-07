"""Core map-to-raceline planning utilities.

This module is intentionally ROS-agnostic. It accepts a 2D occupancy-style map
and produces a closed-loop race line with a speed profile. The high-level
pipeline is:

1) Convert/threshold map data into a clean drivable mask.
2) Extract a closed centerline from track boundaries.
3) Estimate per-point lateral bounds from free-space distance.
4) Optimize lateral offsets to smooth curvature while staying inside bounds.
5) Compute heading and a kinematically-feasible speed profile.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass
class RaceLinePlan:
    """Container for all planning outputs used by downstream nodes."""

    # Centerline in world coordinates (x forward, y left).
    centerline_xy: np.ndarray
    # Optimized race line in world coordinates.
    raceline_xy: np.ndarray
    # Centerline in planner pixel coordinates.
    centerline_px: np.ndarray
    # Race line in planner pixel coordinates.
    raceline_px: np.ndarray
    # Path tangent heading at each race-line point (rad).
    yaw: np.ndarray
    # Reference speed at each race-line point (m/s).
    speed_profile: np.ndarray
    # Per-point signed-offset limits around centerline (m).
    lateral_bounds_m: np.ndarray
    # Solved centerline offsets used to generate raceline (m).
    lateral_offsets_m: np.ndarray
    # Binary drivable mask used for contour extraction.
    drivable_mask: np.ndarray


def wrap_angle(angle: float) -> float:
    """Normalize any angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def occupancy_data_to_gray_image(
    occ_data: np.ndarray,
    width: int,
    height: int,
    occupied_threshold: int = 50,
) -> np.ndarray:
    """Convert ROS OccupancyGrid data into planner grayscale convention.

    Input occupancy values:
    - occupied >= occupied_threshold
    - free == 0
    - unknown (typically -1) or intermediate values

    Output grayscale convention used in this project:
    - 0   -> occupied
    - 127 -> unknown
    - 255 -> free
    """
    grid = np.asarray(occ_data, dtype=np.int16).reshape(height, width)
    # ROS stores row-major from bottom-left; planner images are top-down.
    top_down = np.flipud(grid)

    gray = np.full((height, width), 127, dtype=np.uint8)
    gray[top_down >= occupied_threshold] = 0
    gray[top_down == 0] = 255
    return gray


def gray_image_to_occupancy_data(
    gray: np.ndarray,
    free_thresh: int = 200,
    occ_thresh: int = 90,
) -> np.ndarray:
    """Convert planner grayscale map back to ROS OccupancyGrid values."""
    if gray.ndim != 2:
        raise ValueError('Expected grayscale image')

    gray_i = gray.astype(np.int16)

    # Start with a probabilistic mapping so partially observed boundaries are
    # reflected in occupancy immediately (instead of waiting for hard threshold).
    occ_prob = np.round((255.0 - gray_i.astype(np.float32)) * (100.0 / 255.0)).astype(np.int16)
    occ_prob = np.clip(occ_prob, 0, 100)

    # Clamp clear and fully occupied ends for stability.
    occ_prob[gray_i >= free_thresh] = 0
    occ_prob[gray_i <= occ_thresh] = 100

    occ_top = occ_prob.astype(np.int8)

    # Preserve untouched unknown canvas around neutral gray.
    unknown_mask = np.abs(gray_i - 127) <= 5
    occ_top[unknown_mask] = -1

    # ROS OccupancyGrid expects row-major starting at bottom-left.
    occ_bottom = np.flipud(occ_top)
    return occ_bottom.reshape(-1)


def pixel_to_world(points_px: np.ndarray, scale_px_per_m: float, cx: float, cy: float) -> np.ndarray:
    """Convert planner pixel coordinates to world (x, y) in meters."""
    pts = np.asarray(points_px, dtype=float)
    world = np.empty_like(pts, dtype=float)
    # Pixel y grows downward; world x grows upward in image.
    world[:, 0] = (cy - pts[:, 1]) / scale_px_per_m
    # Pixel x grows right; world y grows right (left-positive car frame style).
    world[:, 1] = (pts[:, 0] - cx) / scale_px_per_m
    return world


def world_to_pixel(points_xy: np.ndarray, scale_px_per_m: float, cx: float, cy: float) -> np.ndarray:
    """Convert world (x, y) in meters into planner pixel coordinates."""
    pts = np.asarray(points_xy, dtype=float)
    px = np.empty_like(pts, dtype=float)
    px[:, 0] = cx + pts[:, 1] * scale_px_per_m
    px[:, 1] = cy - pts[:, 0] * scale_px_per_m
    return px


def _resample_closed_curve(points: np.ndarray, num_samples: int) -> np.ndarray:
    """Resample a closed polyline to uniform arc-length samples."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 4:
        raise ValueError('Not enough points to resample closed curve')

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
        pts = np.vstack([pts, pts[0]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        raise ValueError('Degenerate curve length')

    targets = np.linspace(0.0, total, num_samples, endpoint=False)
    xs = np.interp(targets, s, pts[:, 0])
    ys = np.interp(targets, s, pts[:, 1])
    return np.column_stack([xs, ys])


def _smooth_closed(points: np.ndarray, passes: int = 2) -> np.ndarray:
    """Apply circular three-point smoothing on a closed curve."""
    out = points.copy()
    for _ in range(max(0, passes)):
        out = 0.25 * np.roll(out, 1, axis=0) + 0.5 * out + 0.25 * np.roll(out, -1, axis=0)
    return out


def preprocess_drivable_mask(
    gray: np.ndarray,
    free_thresh: int,
    occ_thresh: int = 90,
    kernel_size: int = 5,
) -> np.ndarray:
    """Create a clean drivable-region mask from grayscale occupancy map."""
    if gray.ndim != 2:
        raise ValueError('Expected grayscale image')

    # Start from high-confidence free/occupied thresholding.
    mask = (gray >= free_thresh).astype(np.uint8) * 255
    obstacle = (gray <= occ_thresh).astype(np.uint8) * 255

    if np.any(obstacle > 0):
        # Slightly dilate obstacles to avoid planning directly on boundaries.
        k_obs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obstacle = cv2.dilate(obstacle, k_obs, iterations=1)
        mask[obstacle > 0] = 0

    # Close small gaps and remove speckle noise.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Keep only the largest connected drivable component.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        raise ValueError('No drivable region found in map')

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest] = 255
    return out


def extract_centerline_from_mask(drivable_mask: np.ndarray, num_samples: int) -> np.ndarray:
    """Extract a closed centerline from drivable mask contours.

    Expected map topology is track-like: one large outer contour and
    (ideally) one inner hole contour. The centerline is approximated by
    pairing each outer sample with its nearest inner sample.
    """
    contours, hierarchy = cv2.findContours(drivable_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        raise ValueError('Could not extract contours from drivable mask')

    h = hierarchy[0]
    outer_idxs = [i for i in range(len(contours)) if h[i, 3] == -1]
    if not outer_idxs:
        raise ValueError('No outer contour found')

    outer_idx = max(outer_idxs, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_idx].reshape(-1, 2).astype(float)

    # Child contours of the selected outer boundary are candidate inner walls.
    child_idxs = [i for i in range(len(contours)) if h[i, 3] == outer_idx]

    outer_rs = _resample_closed_curve(outer, num_samples)

    if child_idxs:
        inner_idx = max(child_idxs, key=lambda i: cv2.contourArea(contours[i]))
        inner = contours[inner_idx].reshape(-1, 2).astype(float)
        inner_rs = _resample_closed_curve(inner, num_samples)
        tree = cKDTree(inner_rs)
        _, nn_idx = tree.query(outer_rs, k=1)
        centerline = 0.5 * (outer_rs + inner_rs[nn_idx])
    else:
        # Fallback if no hole exists: use contour itself
        centerline = outer_rs

    return _smooth_closed(centerline, passes=4)


def _compute_tangent_and_normal(path_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unit tangents and left-pointing unit normals on closed path."""
    d = np.roll(path_xy, -1, axis=0) - np.roll(path_xy, 1, axis=0)
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-6)
    tangent = d / norm
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal


def _estimate_lateral_bounds(
    centerline_px: np.ndarray,
    drivable_mask: np.ndarray,
    scale_px_per_m: float,
    margin_m: float = 0.05,
) -> np.ndarray:
    """Estimate allowable lateral offset around centerline for each sample."""
    # Distance transform gives pixel distance to nearest non-drivable cell.
    dt_px = cv2.distanceTransform(drivable_mask, cv2.DIST_L2, 5)
    h, w = dt_px.shape

    x = np.clip(np.round(centerline_px[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(centerline_px[:, 1]).astype(int), 0, h - 1)

    half_width_m = dt_px[y, x] / scale_px_per_m
    bounds = np.clip(half_width_m - margin_m, 0.05, 1.5)
    return bounds


def optimize_racing_line(
    centerline_xy: np.ndarray,
    lateral_bounds_m: np.ndarray,
    w_curvature: float,
    w_smooth: float,
    w_center_bias: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve bounded lateral-offset smoothing problem for race line.

    Optimization variable is scalar lateral offset e[i] along centerline normal.
    The objective balances:
    - curvature-like second difference penalty
    - first-difference smoothness
    - small centerline bias toward e=0
    """
    n = len(centerline_xy)
    _, normal = _compute_tangent_and_normal(centerline_xy)

    def objective(e: np.ndarray) -> float:
        d1 = np.roll(e, -1) - e
        d2 = np.roll(e, -1) - 2.0 * e + np.roll(e, 1)
        return float(
            w_curvature * np.dot(d2, d2)
            + w_smooth * np.dot(d1, d1)
            + w_center_bias * np.dot(e, e)
        )

    x0 = np.zeros(n, dtype=float)
    # Bound each offset using local free-space estimate.
    bounds = [(-float(b), float(b)) for b in lateral_bounds_m]

    res = minimize(objective, x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 220})
    offsets = res.x if res.success and res.x is not None else x0

    race_xy = centerline_xy + offsets[:, None] * normal
    race_xy = _smooth_closed(race_xy, passes=1)
    return race_xy, offsets


def _compute_yaw(path_xy: np.ndarray) -> np.ndarray:
    """Compute heading from forward differences on closed path."""
    d = np.roll(path_xy, -1, axis=0) - path_xy
    return np.arctan2(d[:, 1], d[:, 0])


def _compute_curvature(path_xy: np.ndarray) -> np.ndarray:
    """Approximate signed curvature kappa = d(heading)/ds on closed path."""
    yaw = _compute_yaw(path_xy)
    yaw_next = np.roll(yaw, -1)
    dyaw = np.arctan2(np.sin(yaw_next - yaw), np.cos(yaw_next - yaw))
    ds = np.linalg.norm(np.roll(path_xy, -1, axis=0) - path_xy, axis=1)
    ds = np.maximum(ds, 1e-3)
    return dyaw / ds


def compute_speed_profile(
    path_xy: np.ndarray,
    v_max: float,
    a_lat_max: float,
    a_long_accel_max: float,
    a_long_brake_max: float,
) -> np.ndarray:
    """Compute speed profile with lateral and longitudinal accel limits.

    Steps:
    1) Apply lateral acceleration bound from curvature.
    2) Forward pass to enforce acceleration limits.
    3) Backward pass to enforce braking limits.
    """
    n = len(path_xy)
    curvature = np.abs(_compute_curvature(path_xy))
    # v <= sqrt(a_lat / |kappa|); clamp very small curvature for stability.
    v_lat = np.sqrt(np.maximum(a_lat_max, 1e-3) / np.maximum(curvature, 1e-3))
    v = np.minimum(v_lat, v_max)

    ds = np.linalg.norm(np.roll(path_xy, -1, axis=0) - path_xy, axis=1)
    ds = np.maximum(ds, 1e-3)

    # Rotate path so the tightest speed point is index 0; this helps closed-loop
    # consistency when applying one-way forward/backward passes.
    start = int(np.argmin(v))
    v_roll = np.roll(v, -start)
    ds_roll = np.roll(ds, -start)

    # Forward pass: acceleration-limited propagation.
    for i in range(1, n):
        v_roll[i] = min(v_roll[i], math.sqrt(max(0.0, v_roll[i - 1] ** 2 + 2.0 * a_long_accel_max * ds_roll[i - 1])))

    # Backward pass: braking-limited propagation.
    for i in range(n - 2, -1, -1):
        v_roll[i] = min(v_roll[i], math.sqrt(max(0.0, v_roll[i + 1] ** 2 + 2.0 * a_long_brake_max * ds_roll[i])))

    return np.roll(v_roll, start)


def reindex_closed_raceline_by_pose(
    raceline_xy: np.ndarray,
    yaw: np.ndarray,
    speed_profile: np.ndarray,
    pose_x: float,
    pose_y: float,
    pose_yaw: float,
    enforce_heading_alignment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate closed-loop indexing so first point is closest to current pose.

    Optionally flips path direction if local heading opposes current vehicle
    heading by > 90 degrees.
    """
    if len(raceline_xy) == 0:
        return raceline_xy, yaw, speed_profile

    pts = np.asarray(raceline_xy, dtype=float)
    heading = np.asarray(yaw, dtype=float)
    speed = np.asarray(speed_profile, dtype=float)

    dxy = pts - np.array([pose_x, pose_y], dtype=float)
    idx = int(np.argmin(np.hypot(dxy[:, 0], dxy[:, 1])))

    if enforce_heading_alignment:
        heading_err = abs(wrap_angle(float(heading[idx]) - pose_yaw))
        if heading_err > (0.5 * math.pi):
            # Reverse the loop if the nearest point faces opposite travel direction.
            pts = pts[::-1].copy()
            speed = speed[::-1].copy()
            heading = _compute_yaw(pts)
            dxy = pts - np.array([pose_x, pose_y], dtype=float)
            idx = int(np.argmin(np.hypot(dxy[:, 0], dxy[:, 1])))

    pts = np.roll(pts, -idx, axis=0)
    speed = np.roll(speed, -idx)
    heading = np.roll(heading, -idx)
    return pts, heading, speed


def plan_from_map(
    gray: np.ndarray,
    scale_px_per_m: float,
    map_center_px_x: float,
    map_center_px_y: float,
    free_thresh: int,
    occ_thresh: int,
    w_curvature: float,
    w_smooth: float,
    w_center_bias: float,
    v_max: float,
    a_lat_max: float,
    a_long_accel_max: float,
    a_long_brake_max: float,
    sample_count: int = 320,
) -> RaceLinePlan:
    """Full map-to-raceline planning pipeline entrypoint."""
    drivable_mask = preprocess_drivable_mask(
        gray,
        free_thresh=free_thresh,
        occ_thresh=occ_thresh,
    )
    centerline_px = extract_centerline_from_mask(drivable_mask, num_samples=sample_count)

    centerline_xy = pixel_to_world(centerline_px, scale_px_per_m, map_center_px_x, map_center_px_y)

    bounds_m = _estimate_lateral_bounds(centerline_px, drivable_mask, scale_px_per_m)

    raceline_xy, offsets_m = optimize_racing_line(
        centerline_xy=centerline_xy,
        lateral_bounds_m=bounds_m,
        w_curvature=w_curvature,
        w_smooth=w_smooth,
        w_center_bias=w_center_bias,
    )

    raceline_px = world_to_pixel(raceline_xy, scale_px_per_m, map_center_px_x, map_center_px_y)
    yaw = _compute_yaw(raceline_xy)
    speed_profile = compute_speed_profile(
        path_xy=raceline_xy,
        v_max=v_max,
        a_lat_max=a_lat_max,
        a_long_accel_max=a_long_accel_max,
        a_long_brake_max=a_long_brake_max,
    )

    return RaceLinePlan(
        centerline_xy=centerline_xy,
        raceline_xy=raceline_xy,
        centerline_px=centerline_px,
        raceline_px=raceline_px,
        yaw=yaw,
        speed_profile=speed_profile,
        lateral_bounds_m=bounds_m,
        lateral_offsets_m=offsets_m,
        drivable_mask=drivable_mask,
    )
