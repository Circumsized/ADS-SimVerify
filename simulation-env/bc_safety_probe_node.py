# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dora-rs==0.3.13",
#     "numpy>=1.26.0",
#     "pyarrow>=14.0.0"
# ]
# ///
import math
import time

import numpy as np
from dora import Node

BEV_WIDTH = 192
BEV_HEIGHT = 192
BEV_METERS_PER_CELL = 20.0 / BEV_WIDTH
BEV_EGO_ROW = (BEV_HEIGHT - 1) * 0.5
BEV_EGO_COL = (BEV_WIDTH - 1) * 0.5
PROBE_PERIOD_SECONDS = 0.20
SAFETY_FORWARD_MIN_M = 0.65
SAFETY_FORWARD_MAX_M = 1.55
SAFETY_CENTER_HALF_WIDTH_M = 0.34
SAFETY_SIDE_HALF_WIDTH_M = 0.90


def bev_cell(forward_m, left_m):
    row = int(round(BEV_EGO_ROW - forward_m / BEV_METERS_PER_CELL))
    col = int(round(BEV_EGO_COL - left_m / BEV_METERS_PER_CELL))
    if 0 <= row < BEV_HEIGHT and 0 <= col < BEV_WIDTH:
        return row, col
    return None


def occupied_count(bev_grid, forward_min, forward_max, left_min, left_max):
    if bev_grid is None:
        return -1
    count = 0
    for forward_m in np.arange(forward_min, forward_max + 1e-6, BEV_METERS_PER_CELL):
        for left_m in np.arange(left_min, left_max + 1e-6, BEV_METERS_PER_CELL):
            cell = bev_cell(forward_m, left_m)
            if cell is None:
                continue
            row, col = cell
            if bev_grid[row, col] > 0:
                count += 1
    return count


def nearest_occupied_distance(bev_grid, forward_min, forward_max, left_min, left_max):
    if bev_grid is None:
        return None
    for forward_m in np.arange(forward_min, forward_max + 1e-6, BEV_METERS_PER_CELL):
        if occupied_count(bev_grid, forward_m, forward_m, left_min, left_max) > 0:
            return float(forward_m)
    return None


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def local_goal(odom, goal):
    dx = goal[0] - odom[0]
    dy = goal[1] - odom[1]
    yaw = odom[2]
    local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return local_x, local_y, math.hypot(local_x, local_y), math.atan2(local_y, local_x)


def fmt(value, precision=3):
    if value is None:
        return "None"
    return f"{value:.{precision}f}"


def main():
    print("[BC Safety Probe] passive probe mounted; no control output will be sent.")
    node = Node()
    goal = [0.52, 4.11]
    odom = None
    prev_odom_sample = None
    cmd = [0.0, 0.0]
    bev_grid = None
    last_bev_time = 0.0
    last_cmd_time = 0.0
    last_print_time = 0.0

    while True:
        event = node.next(timeout=0.01)
        now = time.time()
        if event is None:
            continue
        if event["type"] == "STOP":
            print("[BC Safety Probe] stopped.")
            break
        if event["type"] != "INPUT":
            continue

        ev_id = event["id"]
        data = event["value"].to_numpy()
        if ev_id == "human_prior" and len(data) >= 2:
            goal = [float(data[0]), float(data[1])]
        elif ev_id == "control_cmd" and len(data) >= 2:
            cmd = [float(data[0]), float(data[1])]
            last_cmd_time = now
        elif ev_id == "bev_grid":
            if len(data) == BEV_WIDTH * BEV_HEIGHT:
                bev_grid = data.reshape((BEV_HEIGHT, BEV_WIDTH))
                last_bev_time = now
        elif ev_id == "odometry" and len(data) >= 3:
            new_odom = [
                float(data[0]),
                float(data[1]),
                float(data[2]),
                float(data[3]) if len(data) >= 4 else 0.0,
            ]
            yaw_rate = 0.0
            if prev_odom_sample is not None:
                prev_time, prev_yaw = prev_odom_sample
                dt = now - prev_time
                if dt > 1e-3:
                    yaw_rate = wrap_pi(new_odom[2] - prev_yaw) / dt
            prev_odom_sample = (now, new_odom[2])
            odom = new_odom

            if now - last_print_time < PROBE_PERIOD_SECONDS:
                continue
            last_print_time = now

            local_x, local_y, dist, bearing = local_goal(odom, goal)
            front_m = nearest_occupied_distance(
                bev_grid,
                SAFETY_FORWARD_MIN_M,
                SAFETY_FORWARD_MAX_M,
                -SAFETY_CENTER_HALF_WIDTH_M,
                SAFETY_CENTER_HALF_WIDTH_M,
            )
            center_hits = occupied_count(
                bev_grid,
                SAFETY_FORWARD_MIN_M,
                SAFETY_FORWARD_MAX_M,
                -SAFETY_CENTER_HALF_WIDTH_M,
                SAFETY_CENTER_HALF_WIDTH_M,
            )
            left_hits = occupied_count(
                bev_grid,
                SAFETY_FORWARD_MIN_M,
                SAFETY_FORWARD_MAX_M,
                0.0,
                SAFETY_SIDE_HALF_WIDTH_M,
            )
            right_hits = occupied_count(
                bev_grid,
                SAFETY_FORWARD_MIN_M,
                SAFETY_FORWARD_MAX_M,
                -SAFETY_SIDE_HALF_WIDTH_M,
                0.0,
            )
            if left_hits < 0 or right_hits < 0:
                turn_pref = "unknown"
            elif left_hits > right_hits:
                turn_pref = "right"
            elif right_hits > left_hits:
                turn_pref = "left"
            elif abs(bearing) > 1e-3:
                turn_pref = "goal_sign"
            else:
                turn_pref = "default_left"

            print(
                "[BC SAFETY PROBE] "
                f"age_bev={now - last_bev_time:.2f}s age_cmd={now - last_cmd_time:.2f}s "
                f"pose=({odom[0]:+.2f},{odom[1]:+.2f},{math.degrees(odom[2]):+.1f}deg) "
                f"v_meas={odom[3]:+.3f} yaw_rate={math.degrees(yaw_rate):+.1f}deg/s "
                f"goal_local=({local_x:+.2f},{local_y:+.2f}) dist={dist:.2f} "
                f"bearing={math.degrees(bearing):+.1f}deg "
                f"cmd=({cmd[0]:+.3f},{cmd[1]:+.3f}) "
                f"front_m={fmt(front_m)} hits_center={center_hits} "
                f"hits_left={left_hits} hits_right={right_hits} turn_pref={turn_pref}"
            )


if __name__ == "__main__":
    main()
