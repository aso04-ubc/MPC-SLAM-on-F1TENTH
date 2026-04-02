import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass
class RaceLinePlan:
    centerline_xy: np.ndarray
    raceline_xy: np.ndarray
    centerline_px: np.ndarray
    raceline_px: np.ndarray
    yaw: np.ndarray
    speed_profile: np.ndarray
    lateral_bounds_m: np.ndarray
    lateral_offsets_m: np.ndarray
    drivable_mask: np.ndarray


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def occupancy_data_to_gray_image(
    occ_data: np.ndarray,
    width: int,
    height: int,
    occupied_threshold: int = 50,
) -> np.ndarray:
    """
    Convert OccupancyGrid flattened data (row-major, origin at bottom-left)
    to a top-down grayscale image consistent with mapper/planner pixel convention.
    """
    grid = np.asarray(occ_data, dtype=np.int16).reshape(height, width)
    top_down = np.flipud(grid)

    gray = np.full((height, width), 127, dtype=np.uint8)
    unknown = top_down < 0
    gray[unknown] = 127
    
    known = ~unknown
    gray[known & (top_down >= occupied_threshold)] = 0
    gray[known & (top_down < occupied_threshold)] = 255
    return gray


def gray_image_to_occupancy_data(
    gray: np.ndarray,
    free_thresh: int = 200,
    occ_thresh: int = 90,
) -> np.ndarray:
    """
    Convert top-down grayscale map (0 obstacle / 127 unknown / 255 free)
    into OccupancyGrid flattened int8 data.
    """
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
    pts = np.asarray(points_px, dtype=float)
    world = np.empty_like(pts, dtype=float)
    world[:, 0] = (cy - pts[:, 1]) / scale_px_per_m
    world[:, 1] = (pts[:, 0] - cx) / scale_px_per_m
    return world


def world_to_pixel(points_xy: np.ndarray, scale_px_per_m: float, cx: float, cy: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    px = np.empty_like(pts, dtype=float)
    px[:, 0] = cx + pts[:, 1] * scale_px_per_m
    px[:, 1] = cy - pts[:, 0] * scale_px_per_m
    return px


def _resample_closed_curve(points: np.ndarray, num_samples: int) -> np.ndarray:
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
    if gray.ndim != 2:
        raise ValueError('Expected grayscale image')

    mask = (gray >= free_thresh).astype(np.uint8) * 255
    obstacle = (gray <= occ_thresh).astype(np.uint8) * 255

    if np.any(obstacle > 0):
        k_obs = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        obstacle = cv2.dilate(obstacle, k_obs, iterations=1)
        mask[obstacle > 0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        raise ValueError('No drivable region found in map')

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask, dtype=np.uint8)
    out[labels == largest] = 255
    return out


def extract_centerline_from_mask(
    drivable_mask: np.ndarray,
    num_samples: int,
    centerline_smooth_passes: int = 4,
) -> np.ndarray:
    contours, hierarchy = cv2.findContours(drivable_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        raise ValueError('Could not extract contours from drivable mask')

    h = hierarchy[0]
    outer_idxs = [i for i in range(len(contours)) if h[i, 3] == -1]
    if not outer_idxs:
        raise ValueError('No outer contour found')

    outer_idx = max(outer_idxs, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_idx].reshape(-1, 2).astype(float)

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

    # Smooth the closed loop a little; this is the only smoothing we keep
    # when using the track centerline directly as the reference.
    return _smooth_closed(centerline, passes=centerline_smooth_passes)


def _compute_tangent_and_normal(path_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    dt_px = cv2.distanceTransform(drivable_mask, cv2.DIST_L2, 5)
    h, w = dt_px.shape

    x = np.clip(np.round(centerline_px[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(centerline_px[:, 1]).astype(int), 0, h - 1)

    half_width_m = dt_px[y, x] / scale_px_per_m
    bounds = np.clip(half_width_m - margin_m, 0.05, 1.5)
    return bounds


def _point_is_safe_in_mask(
    point_px: np.ndarray,
    drivable_mask: np.ndarray,
    distance_transform_px: np.ndarray,
    min_distance_px: float,
) -> bool:
    h, w = drivable_mask.shape
    x = int(np.clip(np.round(float(point_px[0])), 0, w - 1))
    y = int(np.clip(np.round(float(point_px[1])), 0, h - 1))
    if drivable_mask[y, x] == 0:
        return False
    return float(distance_transform_px[y, x]) >= float(min_distance_px)


def _project_point_inside_mask(
    centerline_px: np.ndarray,
    candidate_px: np.ndarray,
    drivable_mask: np.ndarray,
    distance_transform_px: np.ndarray,
    min_distance_px: float,
) -> np.ndarray:
    center = np.asarray(centerline_px, dtype=float)
    candidate = np.asarray(candidate_px, dtype=float)

    if _point_is_safe_in_mask(candidate, drivable_mask, distance_transform_px, min_distance_px):
        return candidate

    if not _point_is_safe_in_mask(center, drivable_mask, distance_transform_px, min_distance_px):
        return center

    lo = 0.0
    hi = 1.0
    best = center.copy()

    for _ in range(24):
        mid = 0.5 * (lo + hi)
        test = center + mid * (candidate - center)
        if _point_is_safe_in_mask(test, drivable_mask, distance_transform_px, min_distance_px):
            best = test
            lo = mid
        else:
            hi = mid

    return best


def constrain_raceline_to_mask(
    centerline_px: np.ndarray,
    raceline_px: np.ndarray,
    drivable_mask: np.ndarray,
    min_distance_px: float = 1.0,
) -> np.ndarray:
    if len(centerline_px) != len(raceline_px):
        raise ValueError('centerline_px and raceline_px must have the same length')

    distance_transform_px = cv2.distanceTransform(drivable_mask, cv2.DIST_L2, 5)
    safe_raceline = np.empty_like(raceline_px, dtype=float)

    for idx, (center_pt, race_pt) in enumerate(zip(centerline_px, raceline_px)):
        safe_raceline[idx] = _project_point_inside_mask(
            centerline_px=np.asarray(center_pt, dtype=float),
            candidate_px=np.asarray(race_pt, dtype=float),
            drivable_mask=drivable_mask,
            distance_transform_px=distance_transform_px,
            min_distance_px=min_distance_px,
        )

    return safe_raceline


def optimize_racing_line(
    centerline_xy: np.ndarray,
    lateral_bounds_m: np.ndarray,
    w_curvature: float,
    w_smooth: float,
    w_center_bias: float,
) -> Tuple[np.ndarray, np.ndarray]:
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
    bounds = [(-float(b), float(b)) for b in lateral_bounds_m]

    res = minimize(objective, x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 220})
    offsets = res.x if res.success and res.x is not None else x0

    race_xy = centerline_xy + offsets[:, None] * normal
    race_xy = _smooth_closed(race_xy, passes=1)
    return race_xy, offsets


def _compute_yaw(path_xy: np.ndarray) -> np.ndarray:
    d = np.roll(path_xy, -1, axis=0) - path_xy
    return np.arctan2(d[:, 1], d[:, 0])


def _compute_curvature(path_xy: np.ndarray) -> np.ndarray:
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
    n = len(path_xy)
    curvature = np.abs(_compute_curvature(path_xy))
    v_lat = np.sqrt(np.maximum(a_lat_max, 1e-3) / np.maximum(curvature, 1e-3))
    v = np.minimum(v_lat, v_max)

    ds = np.linalg.norm(np.roll(path_xy, -1, axis=0) - path_xy, axis=1)
    ds = np.maximum(ds, 1e-3)

    start = int(np.argmin(v))
    v_roll = np.roll(v, -start)
    ds_roll = np.roll(ds, -start)

    for i in range(1, n):
        v_roll[i] = min(v_roll[i], math.sqrt(max(0.0, v_roll[i - 1] ** 2 + 2.0 * a_long_accel_max * ds_roll[i - 1])))

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
    use_centerline_raceline: bool = False,
    centerline_smooth_passes: int = 4,
) -> RaceLinePlan:
    drivable_mask = preprocess_drivable_mask(
        gray,
        free_thresh=free_thresh,
        occ_thresh=occ_thresh,
    )
    centerline_px = extract_centerline_from_mask(
        drivable_mask,
        num_samples=sample_count,
        centerline_smooth_passes=centerline_smooth_passes,
    )

    centerline_xy = pixel_to_world(centerline_px, scale_px_per_m, map_center_px_x, map_center_px_y)

    bounds_m = _estimate_lateral_bounds(centerline_px, drivable_mask, scale_px_per_m)

    if use_centerline_raceline:
        # Directly use the track centerline as the reference.
        # This removes the extra optimization/smoothing layer and typically
        # makes the reference line "more obedient" to the track center.
        raceline_xy = centerline_xy.copy()
        raceline_px = centerline_px.copy()
        offsets_m = np.zeros(len(centerline_xy), dtype=float)
    else:
        raceline_xy, offsets_m = optimize_racing_line(
            centerline_xy=centerline_xy,
            lateral_bounds_m=bounds_m,
            w_curvature=w_curvature,
            w_smooth=w_smooth,
            w_center_bias=w_center_bias,
        )

        raceline_px = world_to_pixel(raceline_xy, scale_px_per_m, map_center_px_x, map_center_px_y)
        raceline_px = constrain_raceline_to_mask(
            centerline_px=centerline_px,
            raceline_px=raceline_px,
            drivable_mask=drivable_mask,
            min_distance_px=1.0,
        )
        raceline_xy = pixel_to_world(raceline_px, scale_px_per_m, map_center_px_x, map_center_px_y)
        _, normal = _compute_tangent_and_normal(centerline_xy)
        offsets_m = np.sum((raceline_xy - centerline_xy) * normal, axis=1)

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
