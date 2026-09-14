#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import numpy as np

PHASE4_DIR = Path(__file__).resolve().parents[1] / "phase4"
sys.path.insert(0, str(PHASE4_DIR))
from warehouse_semantics import CHANNELS, FREE_CHANNELS, channel_id  # noqa: E402

OBSTACLE_MIN_HEIGHT_M = 0.02
OBSTACLE_MAX_HEIGHT_M = 0.35
FLOOR_MIN_HEIGHT_M = -0.05
FLOOR_MAX_HEIGHT_M = 0.08


def camera_depth_to_body(semantic_ids, depth, sensor):
    height, width = semantic_ids.shape
    intrinsics = sensor["intrinsics"]
    ext = sensor["body_extrinsics"]
    pixel_v, pixel_u = np.indices((height, width), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)
    safe_depth = np.where(valid, depth, 0.0)
    x_camera = (pixel_u - intrinsics["cx"]) * safe_depth / intrinsics["fx"]
    y_camera = (pixel_v - intrinsics["cy"]) * safe_depth / intrinsics["fy"]

    sr, cr = math.sin(ext["roll_rad"]), math.cos(ext["roll_rad"])
    sp, cp = math.sin(ext["pitch_rad"]), math.cos(ext["pitch_rad"])
    sy, cyaw = math.sin(ext["yaw_rad"]), math.cos(ext["yaw_rad"])
    x_level = cr * x_camera - sr * y_camera
    y_pitched = sr * x_camera + cr * y_camera
    heading_forward = -sp * y_pitched + cp * safe_depth
    heading_left = -x_level
    point_height = ext["height_m"] - (cp * y_pitched + sp * safe_depth)
    relative_forward = cyaw * heading_forward - sy * heading_left
    relative_left = sy * heading_forward + cyaw * heading_left
    return (
        relative_forward + ext["forward_m"],
        relative_left + ext["left_m"],
        point_height,
        valid,
    )


def one_hot(class_ids):
    return np.stack(
        [class_ids == class_id for class_id in range(len(CHANNELS))]
    ).astype(np.uint8)


def depth_lift_semantic_bev(semantic_ids, depth, sensor, bev):
    forward, left, point_height, valid = camera_depth_to_body(
        semantic_ids, depth, sensor
    )
    rows, cols = bev["shape"][0], bev["shape"][1]
    meters = bev["meters_per_cell"]
    ego_row, ego_col = bev["ego_origin_cell"]
    cell_row = np.rint(ego_row - forward / meters).astype(np.int32)
    cell_col = np.rint(ego_col - left / meters).astype(np.int32)
    valid &= (
        (cell_row >= 0)
        & (cell_row < rows)
        & (cell_col >= 0)
        & (cell_col < cols)
    )

    result = np.full(
        (rows, cols), channel_id("unknown_or_unlabeled"), dtype=np.uint8
    )
    observed = np.zeros((rows, cols), dtype=bool)
    class_priority = [13, 0, 1, *range(2, 13)]
    for class_index in class_priority:
        selected = valid & (semantic_ids == class_index)
        if class_index in FREE_CHANNELS:
            selected &= (
                (point_height >= FLOOR_MIN_HEIGHT_M)
                & (point_height <= FLOOR_MAX_HEIGHT_M)
            )
        else:
            selected &= (
                (point_height >= OBSTACLE_MIN_HEIGHT_M)
                & (point_height <= OBSTACLE_MAX_HEIGHT_M)
            )
        result[cell_row[selected], cell_col[selected]] = class_index
        observed[cell_row[selected], cell_col[selected]] = True
    return one_hot(result), observed
