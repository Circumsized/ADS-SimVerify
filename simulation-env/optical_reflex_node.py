# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dora-rs==0.3.13",
#     "numpy>=1.26.0",
#     "opencv-python>=4.8.0",
#     "pyarrow>=14.0.0"
# ]
# ///
import cv2
import numpy as np
import pyarrow as pa
import time
from dora import Node

TTC_SAFE_SECONDS = 10.0
TTC_EMERGENCY_SECONDS = 0.1
TTC_PUBLISH_PERIOD_SECONDS = 0.05
VISUAL_FRAME_STALE_SECONDS = 0.20
CAMERA_FX = 204.25533
CAMERA_FY = 153.19150
CAMERA_CX = 319.5
CAMERA_CY = 239.5
CAMERA_FORWARD_OFFSET_M = 0.06935859
CAMERA_LEFT_OFFSET_M = -0.00000002
CAMERA_HEIGHT_M = 0.13328385
CAMERA_YAW_RAD = 0.076109
CAMERA_PITCH_RAD = 0.168662
CAMERA_ROLL_RAD = 0.0
BEV_METERS_PER_CELL = 20.0 / 192.0
BEV_WIDTH = 192
BEV_HEIGHT = 192
BEV_EGO_ROW = 95.5
BEV_EGO_COL = 95.5

def pixel_to_bev(u, v):
    normalized_x = (u - CAMERA_CX) / CAMERA_FX
    normalized_y = (v - CAMERA_CY) / CAMERA_FY
    sin_roll, cos_roll = np.sin(CAMERA_ROLL_RAD), np.cos(CAMERA_ROLL_RAD)
    sin_pitch, cos_pitch = np.sin(CAMERA_PITCH_RAD), np.cos(CAMERA_PITCH_RAD)
    pitched_y = sin_roll * normalized_x + cos_roll * normalized_y
    denominator = cos_pitch * pitched_y + sin_pitch
    if not np.isfinite(denominator) or denominator <= 1e-6:
        return None
    ray_scale = CAMERA_HEIGHT_M / denominator
    heading_forward = ray_scale * (-sin_pitch * pitched_y + cos_pitch)
    heading_left = -ray_scale * (cos_roll * normalized_x - sin_roll * normalized_y)
    sin_yaw, cos_yaw = np.sin(CAMERA_YAW_RAD), np.cos(CAMERA_YAW_RAD)
    forward_m = cos_yaw * heading_forward - sin_yaw * heading_left + CAMERA_FORWARD_OFFSET_M
    left_m = sin_yaw * heading_forward + cos_yaw * heading_left + CAMERA_LEFT_OFFSET_M
    row = int(round(BEV_EGO_ROW - forward_m / BEV_METERS_PER_CELL))
    col = int(round(BEV_EGO_COL - left_m / BEV_METERS_PER_CELL))
    if 0 <= row < BEV_HEIGHT and 0 <= col < BEV_WIDTH:
        return row, col
    return None

def publish_ttc(dora_node, ttc):
    ttc = float(np.clip(ttc, TTC_EMERGENCY_SECONDS, TTC_SAFE_SECONDS))
    ttc_arrow = pa.array([ttc], type=pa.float32())
    dora_node.send_output("ttc", ttc_arrow)

def main():
    print("========================================================")
    print("Optical Flow Reflex Node Activated")
    print("Features: Epipolar Rotational Blanking | Semantic Masking")
    print("========================================================")
    
    dora_node = Node()
    
    prev_gray = None
    prev_pts = None
    
    # Vehicle motion tracker states
    prev_yaw = None
    prev_time = None
    omega_z = 0.0 # Instantaneous yaw rate
    last_ttc = TTC_EMERGENCY_SECONDS
    last_ttc_publish_time = 0.0
    last_visual_frame_time = 0.0
    
    # Unknown space is treated as occupied until a fresh BEV grid arrives.
    bev_grid = np.full((BEV_HEIGHT, BEV_WIDTH), 255, dtype=np.uint8)
    
    # Camera intrinsic properties (VGA resolution)
    fx = CAMERA_FX
    cx = CAMERA_CX
    cy = CAMERA_CY
    
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=120, qualityLevel=0.05, minDistance=10, blockSize=7)
    
    try:
        while True:
            event = dora_node.next(timeout=0.01)
            if event is None:
                now = time.time()
                if now - last_ttc_publish_time >= TTC_PUBLISH_PERIOD_SECONDS:
                    heartbeat_ttc = last_ttc if now - last_visual_frame_time <= VISUAL_FRAME_STALE_SECONDS else TTC_EMERGENCY_SECONDS
                    publish_ttc(dora_node, heartbeat_ttc)
                    last_ttc_publish_time = now
                continue
                
            ev_type = event["type"]
            if ev_type == "INPUT":
                ev_id = event["id"]
                
                # 1. Update vehicle yaw rate from odometry
                if ev_id == "odometry":
                    odom_data = event["value"].to_numpy()
                    if len(odom_data) >= 3:
                        curr_yaw = float(odom_data[2])
                        curr_time = time.time()
                        
                        if prev_yaw is not None and prev_time is not None:
                            dt = curr_time - prev_time
                            if dt > 0.001:
                                dyaw = curr_yaw - prev_yaw
                                # normalize to [-pi, pi]
                                dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
                                omega_z = dyaw / dt
                                
                        prev_yaw = curr_yaw
                        prev_time = curr_time
                        
                # 2. Update semantic occupancy grid from perception_node
                elif ev_id == "bev_grid":
                    grid_flat = event["value"].to_numpy()
                    if len(grid_flat) == 192 * 192:
                        bev_grid = grid_flat.reshape((192, 192))
                        
                # 3. Main processing loop triggered by new visual frames
                elif ev_id == "jpeg_image":
                    last_visual_frame_time = time.time()
                    jpeg_bytes = event["value"].to_numpy()
                    frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_GRAYSCALE)
                    if frame is None:
                        last_ttc = TTC_EMERGENCY_SECONDS
                        now = time.time()
                        if now - last_ttc_publish_time >= TTC_PUBLISH_PERIOD_SECONDS:
                            publish_ttc(dora_node, last_ttc)
                            last_ttc_publish_time = now
                        continue
                        
                    h, w = frame.shape
                    ttc = TTC_SAFE_SECONDS
                    
                    if prev_gray is None or prev_pts is None or len(prev_pts) < 15:
                        prev_gray = frame
                        prev_pts = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
                    else:
                        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame, prev_pts, None, **lk_params)
                        if next_pts is None or status is None:
                            prev_gray = frame
                            prev_pts = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
                            last_ttc = TTC_EMERGENCY_SECONDS
                            continue
                        
                        good_new = next_pts[status == 1]
                        good_old = prev_pts[status == 1]
                        
                        if len(good_new) > 10:
                            divergence_sum = 0.0
                            valid_points = 0
                            
                            for i, (new, old) in enumerate(zip(good_new, good_old)):
                                u_old, v_old = old[0], old[1]
                                u_new, v_new = new[0], new[1]
                                
                                dx = u_new - u_old
                                dy = v_new - v_old
                                
                                # A. Epipolar Rotational Blanking (Yaw Rate Cancellation)
                                x_rel = u_old - cx
                                y_rel = v_old - cy
                                
                                u_rot = -fx * omega_z - (x_rel**2 / fx) * omega_z
                                v_rot = -(x_rel * y_rel / fx) * omega_z
                                
                                dx_comp = dx - u_rot
                                dy_comp = dy - v_rot
                                
                                # B. Semantic Masking (Verify if point is on road via inverse-homography projection)
                                is_obstacle = True
                                bev_cell = pixel_to_bev(u_old, v_old)
                                if bev_cell is not None:
                                    row, col = bev_cell
                                    if bev_grid[row, col] == 0: # 0 means Road (Free Space)
                                        is_obstacle = False
                                        
                                # C. Cumulative Translation Divergence (Only evaluate true obstacles)
                                if is_obstacle:
                                    r_sq = x_rel**2 + y_rel**2
                                    if r_sq > 400: # filter central singularity zone
                                        div = (dx_comp * x_rel + dy_comp * y_rel) / r_sq
                                        divergence_sum += div
                                        valid_points += 1
                                        
                            if valid_points > 0:
                                mean_divergence = divergence_sum / valid_points
                                if mean_divergence > 0.01: # looming detection threshold
                                    ttc = 1.0 / mean_divergence
                                    
                        prev_gray = frame
                        prev_pts = cv2.goodFeaturesToTrack(frame, mask=None, **feature_params)
                        
                    last_ttc = float(np.clip(ttc, TTC_EMERGENCY_SECONDS, TTC_SAFE_SECONDS))

                now = time.time()
                if now - last_ttc_publish_time >= TTC_PUBLISH_PERIOD_SECONDS:
                    heartbeat_ttc = last_ttc if now - last_visual_frame_time <= VISUAL_FRAME_STALE_SECONDS else TTC_EMERGENCY_SECONDS
                    publish_ttc(dora_node, heartbeat_ttc)
                    last_ttc_publish_time = now
                    
            elif ev_type == "STOP":
                print("\nOptical reflex node unmounted cleanly.")
                break
                
    except Exception as e:
        print(f"Optical Reflex Node Error: {e}")

if __name__ == "__main__":
    main()
