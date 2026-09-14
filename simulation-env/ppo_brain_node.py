import os
import sys
import time
import numpy as np
import onnxruntime as ort
import pyarrow as pa
from collections import deque
from dora import Node
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V_MAX = 0.80
KAPPA_MAX = 1.25
HEADING_RECOVERY_ENTER_RAD = np.deg2rad(60.0)
HEADING_RECOVERY_EXIT_RAD = np.deg2rad(35.0)
HEADING_RECOVERY_W_MAX = 0.90
HEADING_RECOVERY_KP = 0.85

def main():
    print("=========================================================================")
    print("15D Cleaned PPO Neural Brain Deployment Node")
    print("=========================================================================")
    model_path = os.path.join(REPO_ROOT, "model", "spiced_brain.onnx")
    if not os.path.exists(model_path):
        print(f"[-] Error: Target PPO model spiced_brain.onnx not found at {model_path}")
        sys.exit(1)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dora_node = Node()
    state_odom = [0.0, 0.0, 0.0]     
    goal_pose = [0.52, 4.11]         
    history_queue = deque(maxlen=5)
    inference_count = 0
    heading_recovery_active = False
    prev_odom_time = None
    prev_odom_yaw = None
    print("[+] 15D PPO Neural Brain successfully mounted. Entering 100Hz inference loop...")
    print("-" * 73)
    try:
        while True:
            event = dora_node.next(timeout=0.01)
            if event is None:
                continue
            ev_type = event["type"]
            if ev_type == "INPUT":
                ev_id = event["id"]
                if ev_id == "human_prior":
                    prior_arr = event["value"].to_numpy()
                    if len(prior_arr) >= 2:
                        goal_pose = [float(prior_arr[0]), float(prior_arr[1])]
                elif ev_id == "odometry":
                    odom_arr = event["value"].to_numpy()
                    if len(odom_arr) >= 3:
                        measured_v = float(odom_arr[3]) if len(odom_arr) >= 4 else 0.0
                        state_odom = [float(odom_arr[0]), float(odom_arr[1]), float(odom_arr[2]), measured_v]
                    yaw_rate = 0.0
                    now = time.time()
                    if prev_odom_time is not None and prev_odom_yaw is not None:
                        dt = now - prev_odom_time
                        if dt > 1e-3:
                            dyaw = (state_odom[2] - prev_odom_yaw + np.pi) % (2.0 * np.pi) - np.pi
                            yaw_rate = dyaw / dt
                    prev_odom_time = now
                    prev_odom_yaw = state_odom[2]
                    dx = goal_pose[0] - state_odom[0]
                    dy = goal_pose[1] - state_odom[1]
                    yaw = state_odom[2]
                    local_goal_x = dx * np.cos(yaw) + dy * np.sin(yaw)
                    local_goal_y = -dx * np.sin(yaw) + dy * np.cos(yaw)
                    local_goal_dist = np.sqrt(local_goal_x**2 + local_goal_y**2)
                    bearing = np.arctan2(local_goal_y, local_goal_x)
                    current_frame = [
                        local_goal_x * 0.20,
                        local_goal_y * 0.20,
                        local_goal_dist * 0.20
                    ]
                    if len(history_queue) == 0:
                        for _ in range(5):
                            history_queue.append(current_frame)
                    else:
                        history_queue.append(current_frame)
                    state_15 = np.array(list(history_queue), dtype=np.float32).flatten().reshape(1, 15)
                    run_outs = session.run(None, {input_name: state_15})
                    action = run_outs[0][0] 
                    a_vel = float(action[0])
                    a_kappa = float(action[1])
                    if heading_recovery_active:
                        heading_recovery_active = abs(bearing) > HEADING_RECOVERY_EXIT_RAD
                    elif abs(bearing) > HEADING_RECOVERY_ENTER_RAD:
                        heading_recovery_active = True
                    if heading_recovery_active:
                        v_des = 0.0
                        w_ref = float(np.clip(HEADING_RECOVERY_KP * bearing, -HEADING_RECOVERY_W_MAX, HEADING_RECOVERY_W_MAX))
                        control_mode = "heading_recovery"
                    else:
                        v_des = np.clip(a_vel, 0.0, 1.0) * V_MAX
                        kappa = np.clip(a_kappa, -1.0, 1.0) * KAPPA_MAX
                        w_ref = kappa * v_des
                        control_mode = "bc"
                    control_arrow = pa.array([v_des, w_ref], type=pa.float32())
                    dora_node.send_output("control_cmd", control_arrow)
                    inference_count += 1
                    if inference_count % 100 == 0:
                        print(
                            f"[BC Brain] dist={local_goal_dist:.2f}m "
                            f"bearing={np.rad2deg(bearing):+.1f}deg mode={control_mode} "
                            f"action=({a_vel:.3f},{a_kappa:.3f}) "
                            f"cmd=({v_des:.3f},{w_ref:.3f}) "
                            f"v_meas={state_odom[3]:+.3f} yaw_rate={np.rad2deg(yaw_rate):+.1f}deg/s"
                        )
            elif ev_type == "STOP":
                print("\n[PPO Brain Node] Unmounted cleanly.")
                break
    except Exception as e:
        print(f"[-] PPO Brain Node runtime error: {e}")
if __name__ == "__main__":
    main()
