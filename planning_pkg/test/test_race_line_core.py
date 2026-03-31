import numpy as np

from planning_pkg.race_line_core import (
    extract_centerline_from_mask,
    gray_image_to_occupancy_data,
    occupancy_data_to_gray_image,
    pixel_to_world,
    plan_from_map,
    preprocess_drivable_mask,
    reindex_closed_raceline_by_pose,
    world_to_pixel,
)


def make_annulus_map(size=320, outer_r=120, inner_r=75):
    yy, xx = np.indices((size, size))
    cx = size // 2
    cy = size // 2
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    img = np.full((size, size), 127, dtype=np.uint8)
    ring = (rr <= outer_r) & (rr >= inner_r)
    img[ring] = 255
    return img, float(cx), float(cy)


def test_pixel_world_roundtrip():
    pts_px = np.array([[160.0, 160.0], [170.0, 140.0], [120.0, 210.0]], dtype=float)
    xy = pixel_to_world(pts_px, scale_px_per_m=40.0, cx=160.0, cy=160.0)
    back = world_to_pixel(xy, scale_px_per_m=40.0, cx=160.0, cy=160.0)
    assert np.max(np.abs(back - pts_px)) < 1e-6


def test_centerline_extraction_has_closed_loop():
    gray, cx, cy = make_annulus_map()
    mask = preprocess_drivable_mask(gray, free_thresh=200)
    center = extract_centerline_from_mask(mask, num_samples=240)

    assert center.shape == (240, 2)
    step = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    assert float(np.min(step)) > 0.1


def test_race_line_stays_within_lateral_bounds():
    gray, cx, cy = make_annulus_map()

    plan = plan_from_map(
        gray=gray,
        scale_px_per_m=40.0,
        map_center_px_x=cx,
        map_center_px_y=cy,
        free_thresh=200,
        occ_thresh=90,
        w_curvature=30.0,
        w_smooth=8.0,
        w_center_bias=1.0,
        v_max=4.0,
        a_lat_max=3.0,
        a_long_accel_max=2.0,
        a_long_brake_max=3.0,
        sample_count=240,
    )

    assert plan.raceline_xy.shape == (240, 2)
    assert np.all(np.abs(plan.lateral_offsets_m) <= plan.lateral_bounds_m + 1e-6)


def test_speed_profile_respects_vmax():
    gray, cx, cy = make_annulus_map()

    plan = plan_from_map(
        gray=gray,
        scale_px_per_m=40.0,
        map_center_px_x=cx,
        map_center_px_y=cy,
        free_thresh=200,
        occ_thresh=90,
        w_curvature=25.0,
        w_smooth=5.0,
        w_center_bias=1.0,
        v_max=3.5,
        a_lat_max=2.5,
        a_long_accel_max=1.8,
        a_long_brake_max=2.5,
        sample_count=220,
    )

    assert np.all(np.isfinite(plan.speed_profile))
    assert float(np.max(plan.speed_profile)) <= 3.5 + 1e-6
    assert float(np.min(plan.speed_profile)) >= 0.0


def test_occupancy_gray_roundtrip_semantics():
    gray = np.array(
        [
            [255, 127, 0],
            [255, 255, 0],
            [127, 127, 255],
        ],
        dtype=np.uint8,
    )
    occ = gray_image_to_occupancy_data(gray, free_thresh=200, occ_thresh=90)
    restored = occupancy_data_to_gray_image(occ, width=3, height=3, occupied_threshold=50)

    assert restored.shape == gray.shape
    assert int(restored[0, 0]) == 255
    assert int(restored[0, 2]) == 0
    assert int(restored[2, 0]) == 127


def test_pose_reindex_picks_nearest_start():
    t = np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)
    raceline = np.column_stack((2.0 * np.cos(t), 2.0 * np.sin(t)))
    yaw = np.arctan2(np.roll(raceline[:, 1], -1) - raceline[:, 1], np.roll(raceline[:, 0], -1) - raceline[:, 0])
    speed = np.full(len(raceline), 2.0, dtype=float)

    out_xy, out_yaw, out_speed = reindex_closed_raceline_by_pose(
        raceline_xy=raceline,
        yaw=yaw,
        speed_profile=speed,
        pose_x=0.1,
        pose_y=1.95,
        pose_yaw=0.0,
        enforce_heading_alignment=False,
    )

    d0 = np.hypot(out_xy[0, 0] - 0.1, out_xy[0, 1] - 1.95)
    assert float(d0) < 0.35
    assert out_xy.shape == raceline.shape
    assert out_yaw.shape == yaw.shape
    assert out_speed.shape == speed.shape
