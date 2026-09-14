import sys
import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SIM_ENV_DIR = os.path.join(REPO_ROOT, "simulation-env")
ACADOS_CODE_DIR = os.path.join(SIM_ENV_DIR, "c_generated_code")
sys.path.append(SIM_ENV_DIR)
sys.path.append(ACADOS_CODE_DIR)
V_MAX = 0.8
KAPPA_MAX = 1.25
try:
    from acados_template import AcadosOcpSolver
except ImportError:
    AcadosOcpSolver = None
    print("Warning: acados_template could not be loaded. Please ensure acados path is configured.")
class FSDCarGymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}
    def __init__(self, goal_x=0.52, goal_y=4.11, max_steps=150):
        super(FSDCarGymEnv, self).__init__()
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.max_steps = max_steps
        self.current_step = 0
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.state_x = 0.0
        self.state_y = 0.0
        self.state_yaw = 0.0
        self.state_v = 0.0
        self.state_w = 0.0
        self.num_obstacles = 3
        self.obs_init_poses = np.array([
            [0.2, 1.5, 0.0],
            [0.4, 2.8, 0.0],
            [-0.1, 3.5, 0.0]
        ])
        self.obs_poses = self.obs_init_poses.copy()
        self.obs_history = deque(maxlen=5)
        self.prev_dist_to_goal = 0.0
        self.solver = None
        json_path = os.path.join(SIM_ENV_DIR, "acados_ocp.json")
        if os.path.exists(json_path) and AcadosOcpSolver is not None:
            try:
                self.solver = AcadosOcpSolver(None, json_file=json_path, generate=False, build=False)
            except Exception as exc:
                print(f"Warning: acados solver could not be initialized ({exc}). Mock solver fallback active.")
        else:
            print(f"Warning: acados solver config not found at: {json_path}. Mock solver fallback active.")
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_x = 0.0
        self.state_y = 0.0
        self.state_yaw = 0.0
        self.state_v = 0.0
        self.state_w = 0.0
        self.current_step = 0
        self.obs_poses = self.obs_init_poses.copy()
        if self.solver is not None:
            init_state = np.array([0.0, 0.0, 0.0, 0.0])
            self.solver.constraints_set(0, "lbx", init_state)
            self.solver.constraints_set(0, "ubx", init_state)
        start_dx = self.goal_x - self.state_x
        start_dy = self.goal_y - self.state_y
        self.prev_dist_to_goal = math.sqrt(start_dx**2 + start_dy**2)
        self.obs_history.clear()
        first_frame = self._get_single_obs()
        for _ in range(5):
            self.obs_history.append(first_frame)
        return self._get_obs_15(), {}
    def _get_single_obs(self):
        dx = self.goal_x - self.state_x
        dy = self.goal_y - self.state_y
        local_goal_x = dx * math.cos(self.state_yaw) + dy * math.sin(self.state_yaw)
        local_goal_y = -dx * math.sin(self.state_yaw) + dy * math.cos(self.state_yaw)
        local_goal_dist = math.sqrt(local_goal_x**2 + local_goal_y**2)
        return np.array([
            local_goal_x * 0.20,
            local_goal_y * 0.20,
            local_goal_dist * 0.20
        ], dtype=np.float32)
    def _get_obs_15(self):
        return np.concatenate(list(self.obs_history)).astype(np.float32)
    def step(self, action):
        self.current_step += 1
        a_vel = float(action[0])
        a_kappa = float(action[1])
        target_velocity = np.clip(a_vel, 0.0, 1.0) * V_MAX
        kappa = np.clip(a_kappa, -1.0, 1.0) * KAPPA_MAX
        w_ref = kappa * target_velocity
        axis_a = 0.35
        axis_b = 0.25
        Q_y_tuned = 45.0
        nmpc_substeps = 10
        dt_sub = 0.01
        collision_occurred = False
        off_road_occurred = False
        for sub_step in range(nmpc_substeps):
            t_world = (self.current_step * nmpc_substeps + sub_step) * dt_sub
            for idx in range(self.num_obstacles):
                phase = idx * 0.6
                self.obs_poses[idx, 0] = self.obs_init_poses[idx, 0] + 1.2 * math.sin(1.5 * t_world + phase)
            dx = self.goal_x - self.state_x
            dy = self.goal_y - self.state_y
            local_target_x = dx * math.cos(self.state_yaw) + dy * math.sin(self.state_yaw)
            local_target_y = -dx * math.sin(self.state_yaw) + dy * math.cos(self.state_yaw)
            local_target_yaw = np.clip(0.0 - self.state_yaw, -0.25, 0.25)
            target_dist = math.sqrt(local_target_x**2 + local_target_y**2)
            scaled_target_x = local_target_x
            scaled_target_y = local_target_y
            if target_dist > 1.2:
                scale = 1.2 / target_dist
                scaled_target_x *= scale
                scaled_target_y *= scale
            d_ff = scaled_target_x / 3.0
            if self.solver is not None:
                self.solver.constraints_set(0, "lbx", np.array([0.0, 0.0, 0.0, self.state_v]))
                self.solver.constraints_set(0, "ubx", np.array([0.0, 0.0, 0.0, self.state_v]))
                W = np.zeros((6, 6))
                W[0, 0] = 20.0
                W[1, 1] = Q_y_tuned
                W[2, 2] = 5.0
                W[3, 3] = 1.0
                W[4, 4] = 0.1
                W[5, 5] = 0.03
                for k in range(21):
                    t = k / 20.0
                    ref_x = 3.0 * (1.0 - t)**2 * t * d_ff + 3.0 * (1.0 - t) * t**2 * (scaled_target_x - d_ff * math.cos(local_target_yaw)) + t**3 * scaled_target_x
                    ref_y = 3.0 * (1.0 - t) * t**2 * (scaled_target_y - d_ff * math.sin(local_target_yaw)) + t**3 * scaled_target_y
                    ref_yaw = local_target_yaw * (t * t * (3.0 - 2.0 * t)) + (w_ref * t)
                    if k < 20:
                        self.solver.cost_set(k, "W", W)
                        self.solver.set(k, "yref", np.array([ref_x, ref_y, ref_yaw, target_velocity, 0.0, 0.0]))
                    else:
                        W_e = np.diag([20.0, Q_y_tuned, 5.0, 1.0]) * 1.5
                        self.solver.cost_set(k, "W", W_e)
                        self.solver.set(k, "yref", np.array([ref_x, ref_y, ref_yaw, target_velocity]))
                    p = np.array([
                        self.obs_poses[0, 0], self.obs_poses[0, 1], axis_a, axis_b,
                        self.obs_poses[1, 0], self.obs_poses[1, 1], axis_a, axis_b,
                        self.obs_poses[2, 0], self.obs_poses[2, 1], axis_a, axis_b
                    ])
                    for idx in range(3):
                        rel_x = p[4 * idx + 0] - self.state_x
                        rel_y = p[4 * idx + 1] - self.state_y
                        p[4 * idx + 0] = rel_x * math.cos(self.state_yaw) + rel_y * math.sin(self.state_yaw)
                        p[4 * idx + 1] = -rel_x * math.sin(self.state_yaw) + rel_y * math.cos(self.state_yaw)
                    self.solver.set(k, "p", p)
                status = self.solver.solve()
                if status == 0:
                    u_opt = self.solver.get(0, "u")
                    acc = u_opt[0]
                    omega = u_opt[1]
                else:
                    acc, omega = -1.0, 0.0
            else:
                acc = -0.5 * (self.state_v - target_velocity)
                heading_feedback = (local_target_y / (local_target_x**2 + 1e-3)) * 0.3
                omega = np.clip(w_ref + heading_feedback, -1.0, 1.0)
            self.state_x += self.state_v * math.cos(self.state_yaw) * dt_sub
            self.state_y += self.state_v * math.sin(self.state_yaw) * dt_sub
            self.state_yaw += omega * dt_sub
            self.state_v = np.clip(self.state_v + acc * dt_sub, 0.0, 0.8)
            self.state_w = omega
            for obs_pos in self.obs_poses:
                dist = math.sqrt((self.state_x - obs_pos[0])**2 + (self.state_y - obs_pos[1])**2)
                if dist < 0.28:
                    collision_occurred = True
                    break
            if abs(self.state_x) > 1.5:
                off_road_occurred = True
                break
            if collision_occurred or off_road_occurred:
                break
        new_frame = self._get_single_obs()
        self.obs_history.append(new_frame)
        obs_15 = self._get_obs_15()
        sparse_reward = 0.0
        terminated = False
        truncated = False
        dist_to_goal = math.sqrt((self.state_x - self.goal_x)**2 + (self.state_y - self.goal_y)**2)
        if dist_to_goal < 0.15:
            sparse_reward = 10.0
            terminated = True
        elif collision_occurred:
            sparse_reward = -10.0
            terminated = True
        elif off_road_occurred:
            sparse_reward = -10.0
            terminated = True
        elif self.current_step >= self.max_steps:
            sparse_reward = 0.0
            truncated = True
        progress_reward = 3.0 * (self.prev_dist_to_goal - dist_to_goal)
        self.prev_dist_to_goal = dist_to_goal
        smooth_penalty = -0.01 * (a_vel**2 + a_kappa**2)
        total_reward = sparse_reward + progress_reward + smooth_penalty
        return obs_15, total_reward, terminated, truncated, {}
    def close(self):
        if self.solver is not None:
            del self.solver
