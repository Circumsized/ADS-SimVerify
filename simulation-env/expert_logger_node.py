# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dora-rs==0.3.13",
#     "pyarrow>=14.0.0",
#     "numpy>=1.26.0"
# ]
# ///
"""
=================================================================
State logger - 10D platform-agnostic state alignment
Design: dynamic goal subscription and local PAM action normalization
=================================================================
"""
import os
import time
import numpy as np
import pyarrow as pa
from dora import Node

# 标定物理包线限幅 (PAM Limits - 强约束契约，全栈统一)
V_MAX = 0.80       # 最大巡航车速
KAPPA_MAX = 1.25   # 最大期望曲率 rad/m
BEV_WIDTH = 192
BEV_HEIGHT = 192
BEV_METERS_PER_CELL = 20.0 / BEV_WIDTH
BEV_EGO_ROW = (BEV_HEIGHT - 1) * 0.5
BEV_EGO_COL = (BEV_WIDTH - 1) * 0.5
TELEPORT_SPLIT_DIST = 0.75
IDLE_CMD_EPS = 0.02
IDLE_SPEED_EPS = 0.03
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")

def bev_cell(forward_m, left_m):
    row = int(round(BEV_EGO_ROW - forward_m / BEV_METERS_PER_CELL))
    col = int(round(BEV_EGO_COL - left_m / BEV_METERS_PER_CELL))
    if 0 <= row < BEV_HEIGHT and 0 <= col < BEV_WIDTH:
        return row, col
    return None

def nearest_occupied_distance(bev_grid, forward_min, forward_max, left_min, left_max):
    if bev_grid is None:
        return -1.0
    for forward_m in np.arange(forward_min, forward_max + 1e-6, BEV_METERS_PER_CELL):
        for left_m in np.arange(left_min, left_max + 1e-6, BEV_METERS_PER_CELL):
            cell = bev_cell(forward_m, left_m)
            if cell is None:
                continue
            row, col = cell
            if bev_grid[row, col] > 0:
                return float(forward_m)
    return -1.0

def main():
    print("========================================================")
    print("[BlackBox Logger] 10D state logger started")
    print("Features: dynamic goals | PAM normalization | teleport segmentation")
    print("========================================================")
    
    dora_node = Node()
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    state_odom = [0.0, 0.0, 0.0, 0.0]  # x, y, yaw, measured_v
    has_odom = False
    last_odom = None
    latest_bev_grid = None
    
    # 核心解耦：初始化默认目标，随后高频接收慢脑广播的动态 Goal 更新
    current_goal_world = [0.52, 4.11] 
    recording_enabled = True
    
    run_id = 1
    csv_file = None
    record_count = 0
    
    def start_new_run(run_num):
        nonlocal csv_file, record_count
        if csv_file is not None:
            csv_file.close()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(DATASET_DIR, f"spice_run_{run_num:03d}_{timestamp_str}.csv")
        csv_file = open(filename, "w", encoding="utf-8")
        
        # 写入大一统 10 维自洽状态动作舱表头
        csv_file.write(
            "timestamp,"
            "odom_x,odom_y,odom_yaw,"
            "local_goal_x,local_goal_y,local_goal_dist,"
            "current_v,action_v_norm,action_kappa_norm,"
            "cmd_v,cmd_w,"
            "obstacle_front_m,obstacle_left_m,obstacle_right_m\n"
        )
        csv_file.flush()
        record_count = 0
        print(f"\n[BlackBox Logger] Created data segment: Run {run_num:03d} <<<")
        print(f"   -> Output path: {filename}")
        return filename
        
    start_new_run(run_id)
    
    try:
        while True:
            event = dora_node.next(timeout=0.01)
            if event is not None:
                ev_type = event["type"]
                if ev_type == "INPUT":
                    ev_id = event["id"]
                    
                    # 核心重构：动态捕获慢脑广播的最新 Goal 坐标，杜绝空间数据撕裂
                    if ev_id == "human_prior":
                        prior_arr = event["value"].to_numpy()
                        if len(prior_arr) >= 2:
                            current_goal_world = [float(prior_arr[0]), float(prior_arr[1])]
                        if len(prior_arr) >= 3:
                            next_recording_enabled = float(prior_arr[2]) > 0.5
                            if next_recording_enabled != recording_enabled:
                                if not next_recording_enabled:
                                    csv_file.flush()
                                    print("\n[BlackBox Logger] Waiting for manual reposition; control samples paused.")
                                elif record_count > 0:
                                    run_id += 1
                                    start_new_run(run_id)
                                    print("[BlackBox Logger] New start stabilized; opened a new data segment.")
                            recording_enabled = next_recording_enabled
                    elif ev_id == "bev_grid":
                        grid_flat = event["value"].to_numpy()
                        if len(grid_flat) == BEV_WIDTH * BEV_HEIGHT:
                            latest_bev_grid = grid_flat.reshape((BEV_HEIGHT, BEV_WIDTH))
                            
                    elif ev_id == "odometry":
                        data = event["value"].to_numpy()
                        if len(data) >= 3:
                            curr_x, curr_y, curr_yaw = float(data[0]), float(data[1]), float(data[2])
                            measured_v = float(data[3]) if len(data) >= 4 else state_odom[3]
                            
                            # 检测仿真器重置 (拖动小车或 Reset) => 瞬间无感分切文件
                            if recording_enabled and last_odom is not None:
                                dist_jump = np.sqrt((curr_x - last_odom[0])**2 + (curr_y - last_odom[1])**2)
                                if dist_jump > TELEPORT_SPLIT_DIST:
                                    print(f"\n[BlackBox Logger] Vehicle teleport detected (distance: {dist_jump:.2f}m).")
                                    print("   -> Saving the previous segment and switching to a new file...")
                                    run_id += 1
                                    start_new_run(run_id)
                                    
                            state_odom = [curr_x, curr_y, curr_yaw, measured_v]
                            has_odom = True
                            last_odom = (curr_x, curr_y)
                            
                    elif ev_id == "control_cmd":
                        data = event["value"].to_numpy()
                        if len(data) >= 2 and has_odom and recording_enabled:
                            cmd_v, cmd_w = float(data[0]), float(data[1])
                            
                            x_ego, y_ego, yaw_ego = state_odom[0], state_odom[1], state_odom[2]
                            current_v = state_odom[3]
                            if abs(cmd_v) < IDLE_CMD_EPS and abs(cmd_w) < IDLE_CMD_EPS and abs(current_v) < IDLE_SPEED_EPS:
                                continue
                            
                            # 1. 将动态接收到的目标进行车体局部坐标系投影
                            dx = current_goal_world[0] - x_ego
                            dy = current_goal_world[1] - y_ego
                            local_g_x = dx * np.cos(yaw_ego) + dy * np.sin(yaw_ego)
                            local_g_y = -dx * np.sin(yaw_ego) + dy * np.cos(yaw_ego)
                            local_g_dist = np.sqrt(local_g_x**2 + local_g_y**2)
                            
                            # 2. PAM 对齐：就地折算为平台无关的目标速度比与目标曲率
                            action_v_norm = np.clip(cmd_v / V_MAX, 0.0, 1.0)
                            
                            # 避免静止时曲率除零奇异点，使用 eps = 0.01
                            kappa = cmd_w / max(abs(cmd_v), 0.01)
                            action_kappa_norm = np.clip(kappa / KAPPA_MAX, -1.0, 1.0)
                            obstacle_front_m = nearest_occupied_distance(latest_bev_grid, 0.20, 2.00, -0.34, 0.34)
                            obstacle_left_m = nearest_occupied_distance(latest_bev_grid, 0.20, 2.00, 0.00, 0.90)
                            obstacle_right_m = nearest_occupied_distance(latest_bev_grid, 0.20, 2.00, -0.90, 0.00)
                            
                            current_time = time.time()
                            
                            # 3. 10 维全状态舱无损物理落盘
                            csv_file.write(
                                f"{current_time:.4f},"
                                f"{x_ego:.4f},{y_ego:.4f},{yaw_ego:.4f},"
                                f"{local_g_x:.4f},{local_g_y:.4f},{local_g_dist:.4f},"
                                f"{current_v:.4f},{action_v_norm:.4f},{action_kappa_norm:.4f},"
                                f"{cmd_v:.4f},{cmd_w:.4f},"
                                f"{obstacle_front_m:.4f},{obstacle_left_m:.4f},{obstacle_right_m:.4f}\n"
                            )
                            record_count += 1
                            if record_count % 100 == 0:
                                csv_file.flush()
                                print(f"  [Data logger {run_id:03d}] records written {record_count} frames... "
                                      f"(speed ratio: {action_v_norm:.2f} | curvature ratio: {action_kappa_norm:+.2f})")
                                      
                elif ev_type == "STOP":
                    print("\n[BlackBox Logger] DORA stop signal received.")
                    break
    except Exception as e:
        print(f"\nERROR in BlackBox Logger: {e}")
    finally:
        if csv_file is not None:
            csv_file.close()
        print(f"[BlackBox Logger] Logger stopped. Completed {run_id} segments.")

if __name__ == "__main__":
    main()
