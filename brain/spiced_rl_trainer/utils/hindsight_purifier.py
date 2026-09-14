import os
import glob
import csv
import math
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REQUIRED_GOAL_FIELDS = ["local_goal_x", "local_goal_y", "local_goal_dist"]
def row_has_value(row, key):
    return row.get(key) not in (None, "")
def row_float(row, key, default=None):
    value = row.get(key)
    if value in (None, ""):
        if default is None:
            raise KeyError(key)
        return default
    return float(value)
def row_speed(row):
    if row_has_value(row, "current_v"):
        return abs(row_float(row, "current_v"))
    if row_has_value(row, "cmd_v"):
        return abs(row_float(row, "cmd_v"))
    return abs(row_float(row, "action_v_norm", 0.0) * 0.8)
def row_command_speed(row):
    if row_has_value(row, "cmd_v"):
        return abs(row_float(row, "cmd_v"))
    if row_has_value(row, "current_v"):
        return abs(row_float(row, "current_v"))
    return abs(row_float(row, "action_v_norm", 0.0) * 0.8)
def purify_and_hindsight(raw_dir=None):
    if raw_dir is None:
        raw_dir = os.path.join(REPO_ROOT, "dataset")
    purified_dir = os.path.join(raw_dir, "purified")
    os.makedirs(purified_dir, exist_ok=True)
    
    csv_pattern = os.path.join(raw_dir, "spice_run_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print("Error: No raw dataset files found.")
        return

    print("=====================================================================================")
    print("Hindsight Goal Alignment and Purification")
    print("=====================================================================================")
    print(f"{'File Name':<28} | {'Raw':<5} | {'Pauses':<6} | {'Purified':<8} | {'Max Vel':<8} | {'Align'}")
    print("-" * 82)

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        
        # Read all rows into memory
        rows = []
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for name in REQUIRED_GOAL_FIELDS:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                rows.append(row)
                
        if len(rows) < 50:
            # Skip corrupted/too short runs
            print(f"{file_name:<28} | {len(rows):<5} | {'-':<6} | {'0':<8} | {'-':<8} | SKIP (Too short)")
            continue
            
        # 1. Hindsight: Extract the absolute last frame as the Dynamic Goal
        last_row = rows[-1]
        goal_x = row_float(last_row, 'odom_x')
        goal_y = row_float(last_row, 'odom_y')
        
        purified_rows = []
        pause_count = 0
        max_vel = 0.0
        
        # 2. Recalculate homogeneous relative goals for every frame
        for row in rows:
            v = row_speed(row)
            cmd_v = row_command_speed(row)
            odom_x = row_float(row, 'odom_x')
            odom_y = row_float(row, 'odom_y')
            odom_yaw = row_float(row, 'odom_yaw')
            
            dx = goal_x - odom_x
            dy = goal_y - odom_y
            
            # Recalculate ego-centric coordinates based on final pose of this run
            local_goal_x = dx * math.cos(odom_yaw) + dy * math.sin(odom_yaw)
            local_goal_y = -dx * math.sin(odom_yaw) + dy * math.cos(odom_yaw)
            local_goal_dist = math.sqrt(local_goal_x**2 + local_goal_y**2)
            
            if v > max_vel:
                max_vel = v
                
            # Filter out intermediate idle pauses
            if v < 0.05 and local_goal_dist > 0.35 and abs(cmd_v) < 0.01:
                pause_count += 1
                continue
                
            # Update row values
            row['local_goal_x'] = f"{local_goal_x:.6f}"
            row['local_goal_y'] = f"{local_goal_y:.6f}"
            row['local_goal_dist'] = f"{local_goal_dist:.6f}"
            purified_rows.append(row)
            
        if len(purified_rows) < 20:
            print(f"{file_name:<28} | {len(rows):<5} | {pause_count:<6} | {len(purified_rows):<8} | {max_vel:<8.3f} | EMPTY")
            continue
            
        # 3. Write purified, hindsight-aligned file back to disk
        out_path = os.path.join(purified_dir, file_name)
        with open(out_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(purified_rows)
            
        print(f"{file_name:<28} | {len(rows):<5} | {pause_count:<6} | {len(purified_rows):<8} | {max_vel:<8.3f} | SUCCESS (Aligned)")

    print("=====================================================================================")
    print("Hindsight dataset pre-processing completed. Stored at dataset/purified/")

if __name__ == "__main__":
    purify_and_hindsight()
