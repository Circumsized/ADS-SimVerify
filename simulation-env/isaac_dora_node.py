import os
import sys
import struct
import math
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import onnxruntime as ort
import pyarrow as pa
from dora import Node
os.environ["ENABLE_CAMERAS"] = "1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from control_gate import parse_control_command, is_control_fresh
os.environ.setdefault("ISAAC_ASSET_ROOT", os.path.join(REPO_ROOT, "assets"))
BEV_WIDTH = 192
BEV_HEIGHT = 192
BEV_METERS_PER_CELL = 20.0 / BEV_WIDTH
BEV_EGO_ROW = (BEV_HEIGHT - 1) * 0.5
BEV_EGO_COL = (BEV_WIDTH - 1) * 0.5
PHASE2_PROBE_ENABLED = os.environ.get("FSD_PHASE2_PROBE", "0") == "1"
CONTROL_INPUT_ID = os.environ.get("FSD_CONTROL_INPUT", "control_cmd")
CAMERA_IMAGE_WIDTH = 640
CAMERA_IMAGE_HEIGHT = 480
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
BEV_MIN_FORWARD_M = 0.20
GOAL_MARKER_PATH = "/Root/spice_goal_marker"
COMPASS_GOAL_RADIUS = 5.5
MANUAL_REPOSITION_MIN_DIST = 0.75
MANUAL_REPOSITION_STABLE_TICKS = 15
COMPASS_GOAL_DIRECTIONS = [
    ("NW", 135.0),
    ("NW+", 145.0),
    ("WNW", 157.5),
    ("W", 180.0),
    ("W+", 190.0),
    ("WSW", 202.5),
    ("SW+", 215.0),
    ("SW", 225.0),
]

def load_env_manually():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        os.environ[parts[0].strip()] = parts[1].strip()
load_env_manually()
try:
    sys.argv.extend(["--/log/level=error", "--/log/fileLogLevel=error"])
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({
        "headless": False,  
    })
    import omni
    from pxr import Usd, UsdPhysics, PhysxSchema, UsdGeom, Gf
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.experimental.utils.semantics import add_labels
    import omni.replicator.core as rep
    print("NVIDIA Isaac Sim 2026 Core Engine successfully initialized.")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
def ego_to_bev(forward_m, left_m):
    row = int(round(BEV_EGO_ROW - forward_m / BEV_METERS_PER_CELL))
    col = int(round(BEV_EGO_COL - left_m / BEV_METERS_PER_CELL))
    if 0 <= row < BEV_HEIGHT and 0 <= col < BEV_WIDTH:
        return row, col
    return None

def build_probe_ipm_remap():
    map_x = np.full((BEV_HEIGHT, BEV_WIDTH), -1.0, dtype=np.float32)
    map_y = np.full((BEV_HEIGHT, BEV_WIDTH), -1.0, dtype=np.float32)
    sin_yaw, cos_yaw = np.sin(CAMERA_YAW_RAD), np.cos(CAMERA_YAW_RAD)
    sin_pitch, cos_pitch = np.sin(CAMERA_PITCH_RAD), np.cos(CAMERA_PITCH_RAD)
    sin_roll, cos_roll = np.sin(CAMERA_ROLL_RAD), np.cos(CAMERA_ROLL_RAD)
    for row in range(BEV_HEIGHT):
        for col in range(BEV_WIDTH):
            forward_m = (BEV_EGO_ROW - row) * BEV_METERS_PER_CELL
            left_m = (BEV_EGO_COL - col) * BEV_METERS_PER_CELL
            if forward_m < BEV_MIN_FORWARD_M:
                continue
            relative_forward = forward_m - CAMERA_FORWARD_OFFSET_M
            relative_left = left_m - CAMERA_LEFT_OFFSET_M
            heading_forward = cos_yaw * relative_forward + sin_yaw * relative_left
            heading_left = -sin_yaw * relative_forward + cos_yaw * relative_left
            if heading_forward <= 0.0:
                continue
            x_camera_level = -heading_left
            y_camera_pitched = CAMERA_HEIGHT_M * cos_pitch - heading_forward * sin_pitch
            z_camera = CAMERA_HEIGHT_M * sin_pitch + heading_forward * cos_pitch
            if z_camera <= 1e-6:
                continue
            x_camera = cos_roll * x_camera_level + sin_roll * y_camera_pitched
            y_camera = -sin_roll * x_camera_level + cos_roll * y_camera_pitched
            u = CAMERA_FX * x_camera / z_camera + CAMERA_CX
            v = CAMERA_FY * y_camera / z_camera + CAMERA_CY
            if 0.0 <= u < CAMERA_IMAGE_WIDTH and 0.0 <= v < CAMERA_IMAGE_HEIGHT:
                map_x[row, col] = u
                map_y[row, col] = v
    return map_x, map_y

def configure_phase2_semantics(stage, obstacle_paths, car_path):
    obstacle_set = set(obstacle_paths)
    free_count = 0
    obstacle_count = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == "/Root" or path.startswith(car_path):
            continue
        name = prim.GetName().lower()
        try:
            if path in obstacle_set:
                add_labels(prim, labels=["obstacle"], taxonomy="class")
                obstacle_count += 1
            elif any(token in name for token in ["ground", "floor", "plane"]):
                add_labels(prim, labels=["free_space"], taxonomy="class")
                free_count += 1
        except Exception as exc:
            print(f"[Phase2] semantic label skipped for {path}: {exc}")
    print(
        f"[Phase2] semantic labels configured: free_prims={free_count}, "
        f"obstacle_prims={obstacle_count}"
    )

def semantic_gt_to_bev(semantic_data, map_x, map_y):
    if not isinstance(semantic_data, dict):
        return np.full((BEV_HEIGHT, BEV_WIDTH), 255, dtype=np.uint8), False
    ids = np.asarray(semantic_data.get("data"))
    id_to_labels = semantic_data.get("info", {}).get("idToLabels", {})
    if ids.ndim != 2 or ids.shape != (CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH):
        return np.full((BEV_HEIGHT, BEV_WIDTH), 255, dtype=np.uint8), False
    free_ids = []
    for semantic_id, labels in id_to_labels.items():
        label_text = str(labels).lower()
        if "free_space" in label_text and "obstacle" not in label_text:
            try:
                free_ids.append(int(semantic_id))
            except (TypeError, ValueError):
                continue
    if not free_ids or not np.any(np.isin(ids, free_ids)):
        return np.full((BEV_HEIGHT, BEV_WIDTH), 255, dtype=np.uint8), False
    free_mask = np.isin(ids, free_ids).astype(np.uint8) * 255
    bev_free = cv2.remap(
        free_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    occupancy = np.where(bev_free > 0, 0, 255).astype(np.uint8)
    return occupancy, True

def rasterize_usd_obstacles(stage, obstacle_paths, pose_x, pose_y, pose_yaw):
    bev_grid = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cos_yaw = math.cos(pose_yaw)
    sin_yaw = math.sin(pose_yaw)
    for prim_path in obstacle_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_pt = aligned.GetMin()
            max_pt = aligned.GetMax()
            world_corners = [
                (float(min_pt[0]), float(min_pt[1])),
                (float(min_pt[0]), float(max_pt[1])),
                (float(max_pt[0]), float(max_pt[1])),
                (float(max_pt[0]), float(min_pt[1])),
            ]
        except Exception:
            continue
        cells = []
        for world_x, world_y in world_corners:
            dx = world_x - pose_x
            dy = world_y - pose_y
            forward = dx * cos_yaw + dy * sin_yaw
            left = -dx * sin_yaw + dy * cos_yaw
            cell = ego_to_bev(forward, left)
            if cell is not None:
                row, col = cell
                cells.append((col, row))
        if len(cells) >= 3:
            cv2.fillConvexPoly(bev_grid, np.asarray(cells, dtype=np.int32), 255)
    return bev_grid

def leaf_obstacle_paths(obstacle_paths):
    unique_paths = sorted(set(obstacle_paths))
    return [
        path
        for path in unique_paths
        if not any(other.startswith(path + "/") for other in unique_paths)
    ]

def set_goal_marker(stage, goal_xy, marker_path=GOAL_MARKER_PATH, radius=0.18, color=None):
    if color is None:
        color = Gf.Vec3f(1.0, 0.82, 0.0)
    prim = stage.GetPrimAtPath(marker_path)
    if not prim.IsValid():
        sphere = UsdGeom.Sphere.Define(stage, marker_path)
        prim = sphere.GetPrim()
    else:
        sphere = UsdGeom.Sphere(prim)
    sphere.CreateRadiusAttr(float(radius))
    sphere.CreateDisplayColorAttr().Set([color])
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(goal_xy[0]), float(goal_xy[1]), 0.18))

def prim_obstacle_footprint(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xform = UsdGeom.Xformable(prim)
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    center_x, center_y = float(translation[0]), float(translation[1])
    radius = 0.45
    try:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        min_pt = aligned.GetMin()
        max_pt = aligned.GetMax()
        half_x = max(0.0, float(max_pt[0] - min_pt[0]) * 0.5)
        half_y = max(0.0, float(max_pt[1] - min_pt[1]) * 0.5)
        if np.isfinite([half_x, half_y]).all():
            radius = max(radius, math.hypot(half_x, half_y))
    except Exception:
        pass
    return center_x, center_y, radius

def infer_goal_bounds(stage, init_xy):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    candidates = []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if not any(token in name for token in ["ground", "floor", "plane", "racetrack"]):
            continue
        try:
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            min_pt = aligned.GetMin()
            max_pt = aligned.GetMax()
            x_min, x_max = float(min_pt[0]), float(max_pt[0])
            y_min, y_max = float(min_pt[1]), float(max_pt[1])
            if np.isfinite([x_min, x_max, y_min, y_max]).all() and x_max - x_min > 2.0 and y_max - y_min > 2.0:
                candidates.append((x_min, x_max, y_min, y_max))
        except Exception:
            continue
    if candidates:
        return max(candidates, key=lambda b: (b[1] - b[0]) * (b[3] - b[2]))
    x0, y0 = float(init_xy[0]), float(init_xy[1])
    return (x0 - 3.0, x0 + 3.0, y0 - 3.0, y0 + 3.0)

def dedupe_obstacles(points, min_sep=0.20):
    deduped = []
    for point in points:
        merged = False
        for old_idx, old in enumerate(deduped):
            if math.hypot(point[0] - old[0], point[1] - old[1]) < min_sep:
                deduped[old_idx] = (old[0], old[1], max(old[2], point[2]))
                merged = True
                break
        if not merged:
            deduped.append(point)
    return deduped

def goal_inside_bounds(goal_xy, bounds, margin=0.45):
    x_min, x_max, y_min, y_max = bounds
    return x_min + margin <= goal_xy[0] <= x_max - margin and y_min + margin <= goal_xy[1] <= y_max - margin

def goal_clear_of_obstacles(goal_xy, obstacle_footprints):
    return not any(
        math.hypot(goal_xy[0] - ox, goal_xy[1] - oy) < radius + 0.45
        for ox, oy, radius in obstacle_footprints
    )

def build_compass_goals(bounds, init_xy, obstacle_footprints):
    goals = []
    for label, degrees in COMPASS_GOAL_DIRECTIONS:
        angle = math.radians(degrees)
        direction = (math.cos(angle), math.sin(angle))
        selected = None
        for distance in np.linspace(COMPASS_GOAL_RADIUS, 1.5, 9):
            candidate = [
                float(init_xy[0] + direction[0] * distance),
                float(init_xy[1] + direction[1] * distance),
            ]
            if goal_inside_bounds(candidate, bounds) and goal_clear_of_obstacles(candidate, obstacle_footprints):
                selected = candidate
                break
        if selected is None:
            selected = [
                float(init_xy[0] + direction[0] * 1.5),
                float(init_xy[1] + direction[1] * 1.5),
            ]
        goals.append((label, selected))
    return goals

class CLIDDEngine:
    def __init__(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print("CLIDD Local Neural Localization Engine successfully loaded.")
    def extract(self, frame_rgb, top_k=200):
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        pad_y = (640 - 480) // 2
        padded = cv2.copyMakeBorder(gray, pad_y, 640 - 480 - pad_y, 0, 0, cv2.BORDER_CONSTANT, value=0)
        tensor = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0)
        mean, std = tensor.mean(), tensor.std()
        tensor = (tensor - mean) / (std + 1e-6)
        outs = self.session.run(None, {self.input_name: tensor.numpy()})
        desc, scores, rel = [torch.from_numpy(x) for x in outs]
        if desc.shape[1] == 80 and desc.shape[2] == 80:
            desc = desc.permute(0, 3, 1, 2)
        if scores.shape[1] == 80 and scores.shape[2] == 80:
            scores = scores.permute(0, 3, 1, 2)
        if len(rel.shape) == 4 and rel.shape[3] == 1 and rel.shape[1] == 80 and rel.shape[2] == 80:
            rel = rel.permute(0, 3, 1, 2)
        scores = F.softmax(scores, dim=1)[:, :-1, :, :]
        scores = F.pixel_shuffle(scores, 8)
        rel = F.interpolate(rel, size=(640, 640), mode='bilinear', align_corners=False)
        conf = scores * rel
        conf_nms = F.max_pool2d(conf, kernel_size=5, stride=1, padding=2)
        mask = (conf == conf_nms) & (conf > 0.05)
        conf_flat = conf[mask]
        if len(conf_flat) == 0:
            return None
        topk = min(top_k, len(conf_flat))
        scores_k, indices = torch.topk(conf_flat, topk)
        y, x = torch.where(mask[0, 0])
        y, x = y[indices], x[indices]
        grid_x = (x.float() / 639.0) * 2.0 - 1.0
        grid_y = (y.float() / 639.0) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).unsqueeze(2)
        desc_sampled = F.grid_sample(desc, grid, mode='bilinear', align_corners=False)
        desc_sampled = desc_sampled.squeeze(0).squeeze(2).t()
        desc_sampled = F.normalize(desc_sampled, p=2, dim=1)
        y = y.float() - pad_y
        x = x.float()
        valid = (y >= 0) & (y < 480)
        x, y, scores_k, desc_sampled = x[valid], y[valid], scores_k[valid], desc_sampled[valid]
        x_np, y_np, s_np, d_np = x.numpy(), y.numpy(), scores_k.numpy(), desc_sampled.numpy()
        x_arr = pa.array(x_np, type=pa.float32())
        y_arr = pa.array(y_np, type=pa.float32())
        s_arr = pa.array(s_np, type=pa.float32())
        d_flat = pa.array(d_np.flatten(), type=pa.float32())
        d_arr = pa.FixedSizeListArray.from_arrays(d_flat, 64)
        struct_arr = pa.StructArray.from_arrays(
            [x_arr, y_arr, s_arr, d_arr],
            names=['x', 'y', 'score', 'descriptor']
        )
        return struct_arr
def main():
    if "DORA_NODE_CONFIG" not in os.environ:
        print("Standalone simulation testing is bypassed.")
        simulation_app.close()
        sys.exit(0)
    print("Launching FSD local zero-copy integration pipeline with ARS...")
    dora_node = Node()
    fsd_assets_dir = os.path.join(REPO_ROOT, "assets")
    clean_usd_name = "fsd_car_clean1.usd"
    default_usd_name = "fsd_car_racetrack.usd"
    clean_usd_path = os.path.join(fsd_assets_dir, clean_usd_name)
    default_usd_path = os.path.join(fsd_assets_dir, default_usd_name)
    usd_path = clean_usd_path if os.path.exists(clean_usd_path) else default_usd_path
    if not os.path.exists(usd_path):
        sys.exit(1)
    open_stage(usd_path=usd_path)
    world = World(
        physics_dt=0.01, 
        rendering_dt=0.01, 
        backend="numpy"
    )
    from isaacsim.core.rendering_manager import RenderingManager
    RenderingManager.set_dt(0.01)
    car_path = "/Root/jetbot"
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(car_path):
        sys.exit(1)
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(car_path) and prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(1e5)
            physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)
            physx_joint.CreateMaxJointVelocityAttr().Set(100000.0) 
    car = Articulation(prim_paths_expr=car_path, name="jetbot")
    world.scene.add(car)
    world.reset()
    init_positions, init_orientations = car.get_world_poses()
    print(f"Spatial anchoring: saved initial pose: {init_positions[0]}")
    obstacle_paths = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if "jetbot" in prim_path.lower() or "groundplane" in prim_path.lower() or "camera" in prim_path.lower() or "light" in prim_path.lower() or prim_path == "/Root":
            continue
        name = prim.GetName().lower()
        if any(kw in name for kw in ["obstacle", "pallet", "box", "can", "bottle", "sphere", "cone", "table", "chair", "cube", "cylinder"]):
            if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh):
                obstacle_paths.append(prim_path)
    obstacle_paths = leaf_obstacle_paths(obstacle_paths)
    init_xy = [float(init_positions[0][0]), float(init_positions[0][1])]
    map_bounds = infer_goal_bounds(stage, init_xy)
    static_obstacle_footprints = dedupe_obstacles(
        [footprint for path in obstacle_paths if (footprint := prim_obstacle_footprint(stage, path)) is not None]
    )
    print(
        "Compass-goal planner: "
        f"bounds=({map_bounds[0]:.2f},{map_bounds[1]:.2f})x({map_bounds[2]:.2f},{map_bounds[3]:.2f}), "
        f"obstacles={len(static_obstacle_footprints)}"
    )
    from isaacsim.core.prims import XFormPrim
    obstacle_path = "/Root/dynamic_obstacles*/box_obstacle_*"
    obstacle = None
    obs_init_poses = None
    obs_init_rots = None
    if stage.GetPrimAtPath("/Root"):
        obstacle = XFormPrim(prim_paths_expr=obstacle_path, name="box_obstacle")
        world.scene.add(obstacle)
    camera_path = f"{car_path}/chassis/rgb_camera/jetbot_camera"
    semantic_gt_annotator = None
    probe_map_x = None
    probe_map_y = None
    if PHASE2_PROBE_ENABLED:
        configure_phase2_semantics(stage, obstacle_paths, car_path)
        probe_map_x, probe_map_y = build_probe_ipm_remap()
    render_product = rep.create.render_product(camera_path, (640, 480))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    if PHASE2_PROBE_ENABLED:
        semantic_gt_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False},
        )
        semantic_gt_annotator.attach([render_product])
        print("[Phase2] A/B/C probe sources enabled; control path remains unchanged.")
    clidd_model_path = os.path.join(REPO_ROOT, "model", "xfeat_640x640.onnx")
    if not os.path.exists(clidd_model_path):
        sys.exit(1)
    clidd_engine = CLIDDEngine(clidd_model_path)
    if obstacle is not None:
        obs_init_poses, obs_init_rots = obstacle.get_world_poses()
    goal_obstacle_footprints = list(static_obstacle_footprints)
    if obs_init_poses is not None:
        goal_obstacle_footprints.extend((float(pos[0]), float(pos[1]), 0.70) for pos in obs_init_poses)
        goal_obstacle_footprints = dedupe_obstacles(goal_obstacle_footprints)
    compass_goals = build_compass_goals(map_bounds, init_xy, goal_obstacle_footprints)
    for goal_idx, (_, goal_xy) in enumerate(compass_goals):
        set_goal_marker(
            stage,
            goal_xy,
            marker_path=f"{GOAL_MARKER_PATH}_{goal_idx}",
            radius=0.08,
            color=Gf.Vec3f(0.35, 0.35, 0.35),
        )
    print(f"Generated compass goals: {len(compass_goals)}: " + ", ".join(
        f"{label}=({goal[0]:.2f},{goal[1]:.2f})" for label, goal in compass_goals
    ))
    left_idx = 0
    right_idx = 1
    for idx, name in enumerate(car.dof_names):
        if "left" in name.lower():
            left_idx = idx
        elif "right" in name.lower():
            right_idx = idx
    world.play()
    left_sign = 1.0
    right_sign = 1.0
    for step_idx in range(60):
        if step_idx < 30:
            test_targets = np.zeros(len(car.dof_names))
            test_targets[left_idx] = 2.0
            test_targets[right_idx] = 2.0
            car.set_joint_velocity_targets(test_targets)
        elif step_idx == 30:
            joint_vels = car.get_joint_velocities()
            if joint_vels is not None and len(joint_vels) > 0:
                raw_left = float(joint_vels[0][left_idx])
                raw_right = float(joint_vels[0][right_idx])
                left_sign = 1.0 if raw_left >= 0.0 else -1.0
                right_sign = 1.0 if raw_right >= 0.0 else -1.0
            stop_targets = np.zeros(len(car.dof_names))
            car.set_joint_velocity_targets(stop_targets)
        world.step(render=True)
    L = 0.1125
    R = 0.03362 
    tick = 0
    v_cmd = 0.0
    w_cmd = 0.0
    rgb_frame = None
    alpha_ars = 30.0 
    v_left_ref = 0.0
    v_right_ref = 0.0
    current_goal_idx = 0
    current_goal_label, current_goal_pose = compass_goals[current_goal_idx]
    collecting_active = True
    waiting_for_manual_reposition = False
    reposition_anchor = None
    last_wait_pose = None
    reposition_stable_ticks = 0
    set_goal_marker(stage, current_goal_pose)
    dora_node.send_output("human_prior", pa.array([current_goal_pose[0], current_goal_pose[1], 1.0], type=pa.float32()))
    print(
        f"Current compass goal #{current_goal_idx + 1}/{len(compass_goals)} "
        f"{current_goal_label}: ({current_goal_pose[0]:.2f}, {current_goal_pose[1]:.2f})"
    )
    last_valid_control_time = None
    try:
        while simulation_app.is_running():
            world.step(render=True)
            tick += 1
            if obstacle is not None and obs_init_poses is not None:
                t = tick * 0.01
                new_poses = obs_init_poses.copy()
                num_obstacles = len(obs_init_poses)
                phase_offsets = np.arange(num_obstacles) * 0.6
                new_poses[:, 0] = obs_init_poses[:, 0] + 1.2 * np.sin(1.5 * t + phase_offsets)
                obstacle.set_world_poses(positions=new_poses, orientations=obs_init_rots)
            rgb_raw = rgb_annotator.get_data()
            if rgb_raw is not None:
                rgb_frame = rgb_raw
                if len(rgb_frame.shape) == 3 and rgb_frame.shape[2] == 4:
                    rgb_frame = rgb_frame[:, :, :3]
                if rgb_frame.dtype == np.float32 or rgb_frame.dtype == np.float64:
                    rgb_frame = (rgb_frame * 255.0).astype(np.uint8)
            if rgb_frame is not None and rgb_frame.size > 0:
                positions, orientations = car.get_world_poses()
                pose_x, pose_y = float(positions[0][0]), float(positions[0][1])
                qw, qx, qy, qz = orientations[0]
                pose_yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
                reset_this_tick = False
                dx_goal = current_goal_pose[0] - pose_x
                dy_goal = current_goal_pose[1] - pose_y
                dist_to_goal = np.sqrt(dx_goal**2 + dy_goal**2)
                if waiting_for_manual_reposition:
                    car.set_joint_velocity_targets(np.zeros(len(car.dof_names)))
                    v_cmd, w_cmd = 0.0, 0.0
                    v_left_ref, v_right_ref = 0.0, 0.0
                    wait_pose = (pose_x, pose_y)
                    moved_dist = math.hypot(wait_pose[0] - reposition_anchor[0], wait_pose[1] - reposition_anchor[1])
                    frame_move = math.hypot(wait_pose[0] - last_wait_pose[0], wait_pose[1] - last_wait_pose[1])
                    if moved_dist >= MANUAL_REPOSITION_MIN_DIST and frame_move < 0.02:
                        reposition_stable_ticks += 1
                    else:
                        reposition_stable_ticks = 0
                    last_wait_pose = wait_pose
                    if reposition_stable_ticks >= MANUAL_REPOSITION_STABLE_TICKS:
                        waiting_for_manual_reposition = False
                        collecting_active = True
                        last_valid_control_time = time.monotonic()
                        dora_node.send_output("human_prior", pa.array([current_goal_pose[0], current_goal_pose[1], 1.0], type=pa.float32()))
                        print(
                            f"Manual reposition stabilized; collection resumed."
                            f" Current goal #{current_goal_idx + 1}/{len(compass_goals)} {current_goal_label}: "
                            f"({current_goal_pose[0]:.2f}, {current_goal_pose[1]:.2f})"
                        )
                    reset_this_tick = True
                elif dist_to_goal < 0.25:
                    print("\n[Collection segment] Goal reached; recording and chassis control paused.")
                    print("   -> Move the vehicle to a more complex starting point in Isaac Sim; collection resumes when stable.")
                    car.set_joint_velocity_targets(np.zeros(len(car.dof_names)))
                    v_cmd, w_cmd = 0.0, 0.0
                    v_left_ref, v_right_ref = 0.0, 0.0
                    current_goal_idx = (current_goal_idx + 1) % len(compass_goals)
                    current_goal_label, current_goal_pose = compass_goals[current_goal_idx]
                    set_goal_marker(stage, current_goal_pose)
                    collecting_active = False
                    waiting_for_manual_reposition = True
                    reposition_anchor = (pose_x, pose_y)
                    last_wait_pose = reposition_anchor
                    reposition_stable_ticks = 0
                    dora_node.send_output("human_prior", pa.array([current_goal_pose[0], current_goal_pose[1], 0.0], type=pa.float32()))
                    print(
                        f"Next compass goal #{current_goal_idx + 1}/{len(compass_goals)} "
                        f"{current_goal_label}: ({current_goal_pose[0]:.2f}, {current_goal_pose[1]:.2f})"
                    )
                    reset_this_tick = True
                measured_v = 0.0
                if not reset_this_tick:
                    joint_vels = car.get_joint_velocities()
                    if joint_vels is not None and len(joint_vels) > 0:
                        v_left_actual = float(joint_vels[0][left_idx]) * left_sign
                        v_right_actual = float(joint_vels[0][right_idx]) * right_sign
                        measured_v = 0.5 * R * (v_left_actual + v_right_actual)
                arrow_odom = pa.array([pose_x, pose_y, pose_yaw, measured_v], type=pa.float32())
                frame_metadata = {
                    "source_frame_id": int(tick),
                    "sim_time_s": float(tick * 0.01),
                }
                if PHASE2_PROBE_ENABLED:
                    dora_node.send_output("odometry", arrow_odom, metadata=frame_metadata)
                else:
                    dora_node.send_output("odometry", arrow_odom)
                if PHASE2_PROBE_ENABLED:
                    bev_grid = rasterize_usd_obstacles(
                        stage, obstacle_paths, pose_x, pose_y, pose_yaw
                    )
                else:
                    bev_grid = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
                    for yw_sample in np.linspace(pose_y, pose_y + 8.0, 50):
                        for world_x in (-1.5, 1.5):
                            dx = world_x - pose_x
                            dy = yw_sample - pose_y
                            forward = dx * np.cos(pose_yaw) + dy * np.sin(pose_yaw)
                            left = -dx * np.sin(pose_yaw) + dy * np.cos(pose_yaw)
                            cell = ego_to_bev(forward, left)
                            if cell is not None:
                                row, col = cell
                                cv2.circle(bev_grid, (col, row), 4, 255, -1)
                    for prim_path in obstacle_paths:
                        prim = stage.GetPrimAtPath(prim_path)
                        if not prim.IsValid():
                            continue
                        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        )
                        translation = matrix.ExtractTranslation()
                        dx = float(translation[0]) - pose_x
                        dy = float(translation[1]) - pose_y
                        forward = dx * np.cos(pose_yaw) + dy * np.sin(pose_yaw)
                        left = -dx * np.sin(pose_yaw) + dy * np.cos(pose_yaw)
                        cell = ego_to_bev(forward, left)
                        if cell is not None:
                            row, col = cell
                            cv2.circle(bev_grid, (col, row), 8, 255, -1)
                bev_flat = bev_grid.flatten()
                arrow_bev = pa.array(bev_flat, type=pa.uint8())
                if PHASE2_PROBE_ENABLED:
                    dora_node.send_output(
                        "bev_grid",
                        arrow_bev,
                        metadata={**frame_metadata, "source_kind": "usd_oracle"},
                    )
                    semantic_raw = semantic_gt_annotator.get_data()
                    semantic_bev, semantic_valid = semantic_gt_to_bev(
                        semantic_raw,
                        probe_map_x,
                        probe_map_y,
                    )
                    dora_node.send_output(
                        "bev_grid_semantic_gt",
                        pa.array(semantic_bev.flatten(), type=pa.uint8()),
                        metadata={
                            **frame_metadata,
                            "source_kind": "isaac_semantic_gt_ipm",
                            "valid": bool(semantic_valid),
                        },
                    )
                    dora_node.send_output(
                        "semantic_gt_valid",
                        pa.array([int(semantic_valid)], type=pa.uint8()),
                        metadata=frame_metadata,
                    )
                else:
                    dora_node.send_output("bev_grid", arrow_bev)
                clean_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                _, jpeg_encoded = cv2.imencode('.jpg', clean_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                arrow_jpeg = pa.array(jpeg_encoded.flatten(), type=pa.uint8())
                if PHASE2_PROBE_ENABLED:
                    dora_node.send_output(
                        "jpeg_image",
                        arrow_jpeg,
                        metadata={**frame_metadata, "source_kind": "isaac_rgb"},
                    )
                else:
                    dora_node.send_output("jpeg_image", arrow_jpeg)
                if tick % 100 == 0:
                    arrow_clidd_features = clidd_engine.extract(rgb_frame, top_k=200)
                    if arrow_clidd_features is not None:
                        dora_node.send_output("xfeat_features", arrow_clidd_features)
                if tick % 10 == 0:
                    active_flag = 1.0 if collecting_active else 0.0
                    dora_node.send_output("human_prior", pa.array([current_goal_pose[0], current_goal_pose[1], active_flag], type=pa.float32()))
            event = dora_node.next(timeout=0.001)
            if event is not None:
                ev_type = event["type"]
                if ev_type == "INPUT":
                    ev_id = event["id"]
                    if ev_id == CONTROL_INPUT_ID:
                        parsed = parse_control_command(event["value"].to_numpy())
                        if parsed is None:
                            v_cmd, w_cmd = 0.0, 0.0
                            print(f"[Safety] rejected invalid {CONTROL_INPUT_ID}; commanding zero")
                        elif waiting_for_manual_reposition:
                            v_cmd, w_cmd = 0.0, 0.0
                            last_valid_control_time = time.monotonic()
                        else:
                            v_cmd, w_cmd = parsed
                            last_valid_control_time = time.monotonic()
                elif ev_type == "STOP":
                    break
            if not is_control_fresh(last_valid_control_time, time.monotonic()):
                v_cmd, w_cmd = 0.0, 0.0
            v_left_cmd = (v_cmd - w_cmd * L / 2.0) / R
            v_right_cmd = (v_cmd + w_cmd * L / 2.0) / R
            joint_vels = car.get_joint_velocities()
            if joint_vels is not None and len(joint_vels) > 0:
                v_left_actual = float(joint_vels[0][left_idx]) * left_sign
                v_right_actual = float(joint_vels[0][right_idx]) * right_sign
            else:
                v_left_actual = 0.0
                v_right_actual = 0.0
            v_left_ref += 0.01 * alpha_ars * (v_left_cmd - v_left_ref)
            v_right_ref += 0.01 * alpha_ars * (v_right_cmd - v_right_ref)
            u_left = v_left_ref
            u_right = v_right_ref
            targets = np.zeros(len(car.dof_names))
            targets[left_idx] = u_left * left_sign
            targets[right_idx] = u_right * right_sign
            car.set_joint_velocity_targets(targets)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if semantic_gt_annotator is not None:
                semantic_gt_annotator.detach([render_product])
            rgb_annotator.detach([render_product])
            car.set_joint_velocity_targets(np.zeros(len(car.dof_names)))
            world.step(render=True)
            world.stop()
        except Exception:
            pass
        simulation_app.close()
if __name__ == "__main__":
    main()
