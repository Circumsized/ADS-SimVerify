#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contracts/phase4"))
sys.path.insert(0, str(ROOT / "contracts/phase5"))
from oracle_nmpc_closed_loop import DT, OracleGrid, Polyline, astar, load_solver, prune_path, wrap_angle  # noqa: E402
from phase5b_shadow_replay import control_roi, occupancy_from_semantic  # noqa: E402
from phase5c3_candidate_shadow import WarehouseCandidate, candidate_depth_lift  # noqa: E402
from phase5f_dual_nmpc_shadow import bev_obstacle_parameters, solve_shadow  # noqa: E402
from phase5g_controlled_takeover import (  # noqa: E402
    DYNAMIC_HALF_EXTENT_M,
    draw_trajectory,
    dynamic_pose,
    integrate_state,
    obb_aabb_collision,
    scenarios,
    supervise_swept_step,
    write_bev_evidence,
)

PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
JETBOT_SOURCE = ROOT / "assets/fsd_car_racetrack.usd"
MODEL_MANIFEST = ROOT / "model/warehouse_nav14_candidate.json"
ROBOT_PATH = "/Phase5HJetbot"
CAMERA_PATH = f"{ROBOT_PATH}/chassis/rgb_camera/jetbot_camera"
WHEEL_RADIUS_M = 0.03362
WHEEL_BASE_M = 0.1125
PHYSICS_DT = 0.01
PHYSICS_STEPS_PER_CONTROL = 5
CSV_FIELDS = [
    "step", "time_s", "x_m", "y_m", "z_m", "yaw_rad", "velocity_mps",
    "acceleration_mps2", "omega_radps", "target_velocity_mps",
    "left_target_radps", "right_target_radps", "left_measured_radps",
    "right_measured_radps", "goal_distance_m", "goal_yaw_error_rad",
    "path_progress_m", "path_error_m", "candidate_valid_ratio",
    "candidate_obstacle_count", "candidate_minimum_h", "solver_status",
    "render_ms", "inference_ms", "depth_lift_ms", "nmpc_ms",
    "candidate_pipeline_ms", "sensor_to_wheel_ms", "dynamic_x_m",
    "dynamic_y_m", "dynamic_center_distance_m", "static_collision",
    "dynamic_collision", "supervisor_decision", "wheel_command_applied",
    "physics_feedback", "candidate_controls_articulation",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaled_sensor_geometry(sensor, width=320, height=240):
    result = deepcopy(sensor)
    source_width, source_height = sensor["image_size"]
    scale_x, scale_y = width / source_width, height / source_height
    result["image_size"] = [width, height]
    for name in ("fx", "cx"):
        result["intrinsics"][name] *= scale_x
    for name in ("fy", "cy"):
        result["intrinsics"][name] *= scale_y
    return result


def differential_wheel_targets(linear_speed, angular_speed):
    left = (linear_speed - angular_speed * WHEEL_BASE_M * 0.5) / WHEEL_RADIUS_M
    right = (linear_speed + angular_speed * WHEEL_BASE_M * 0.5) / WHEEL_RADIUS_M
    return float(left), float(right)


def quaternion_from_yaw(yaw):
    return np.asarray([[math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]], dtype=np.float32)


def articulation_state(car, left_index, right_index):
    positions, orientations = car.get_world_poses()
    qw, qx, qy, qz = [float(value) for value in orientations[0]]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    joint = car.get_joint_velocities()
    left = float(joint[0][left_index])
    right = float(joint[0][right_index])
    velocity = float(np.clip(0.5 * WHEEL_RADIUS_M * (left + right), 0.0, 0.8))
    return np.asarray([positions[0][0], positions[0][1], yaw, velocity], dtype=np.float64), float(positions[0][2]), left, right


def percentile(rows, field, quantile=0.95):
    return float(np.percentile([float(row[field]) for row in rows], quantile * 100.0))


def reset_robot(car, world, scenario, dynamic):
    zero_targets = np.zeros(len(car.dof_names), dtype=np.float32)
    car.set_joint_velocity_targets(zero_targets)
    car.set_world_poses(
        np.asarray([[scenario.start[0], scenario.start[1], 0.05]], dtype=np.float32),
        quaternion_from_yaw(scenario.start[2]),
    )
    car.set_joint_velocities(np.zeros((1, len(car.dof_names)), dtype=np.float32))
    car.set_velocities(np.zeros((1, 6), dtype=np.float32))
    dynamic.set_world_poses(
        positions=np.asarray([[1000.0, 1000.0, DYNAMIC_HALF_EXTENT_M]], dtype=np.float32),
        orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    for _ in range(20):
        world.step(render=False)


def run_scenario(scenario, grid, solver, model, sensor, bev, fixed_roi, world, car, dynamic, rgb, depth, output, max_steps_override=None):
    raw_path = astar(grid, scenario.start[:2], scenario.goal[:2])
    planned = prune_path(grid, raw_path)
    path = Polyline(planned)
    left_index = car.dof_names.index("left_wheel_joint")
    right_index = car.dof_names.index("right_wheel_joint")
    reset_robot(car, world, scenario, dynamic)
    initial_state, _, _, _ = articulation_state(car, left_index, right_index)
    previous_state = initial_state.copy()
    root_path_length = 0.0
    wheel_odometry_distance = 0.0
    previous_progress = 0.0
    rows = []
    abort_reason = None
    reached = False
    closest_dynamic = float("inf")
    dynamic_encounter_frames = 0
    max_steps = max_steps_override or max(900, int(math.ceil(path.length / 0.10 / DT)))
    telemetry_path = output / f"{scenario.name}.csv"
    with telemetry_path.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for step in range(max_steps):
            time_s = step * DT
            pre_state, _, _, _ = articulation_state(car, left_index, right_index)
            pre_goal_distance = math.hypot(scenario.goal[0] - pre_state[0], scenario.goal[1] - pre_state[1])
            pre_goal_yaw_error = wrap_angle(scenario.goal[2] - pre_state[2])
            if pre_goal_distance < 0.15 and abs(pre_goal_yaw_error) < 0.12 and pre_state[3] < 0.05:
                reached = True
                break
            dynamic_x, dynamic_y = dynamic_pose(scenario.name, time_s)
            dynamic.set_world_poses(
                positions=np.asarray([[dynamic_x, dynamic_y, DYNAMIC_HALF_EXTENT_M]], dtype=np.float32),
                orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            sensor_started = time.perf_counter_ns()
            render_started = time.perf_counter_ns()
            world.step(render=True)
            render_ms = (time.perf_counter_ns() - render_started) / 1e6
            state, z_m, left_measured, right_measured = articulation_state(car, left_index, right_index)
            progress = max(previous_progress, path.nearest_s(state[0], state[1]))
            previous_progress = progress
            goal_distance = math.hypot(scenario.goal[0] - state[0], scenario.goal[1] - state[1])
            goal_yaw_error = wrap_angle(scenario.goal[2] - state[2])
            rgb_data = np.asarray(rgb.get_data())
            depth_data = np.asarray(depth.get_data(), dtype=np.float32)
            if rgb_data.size == 0 or depth_data.size == 0:
                abort_reason = "abort_empty_sensor"
                break
            decoded = cv2.cvtColor(rgb_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
            pipeline_started = time.perf_counter_ns()
            classes, inference_ms = model.infer(decoded)
            depth_started = time.perf_counter_ns()
            semantic, observed = candidate_depth_lift(classes, depth_data, sensor, bev)
            dummy = np.full(depth_data.shape, 13, dtype=np.uint8)
            _, depth_reference = candidate_depth_lift(dummy, depth_data, sensor, bev)
            depth_lift_ms = (time.perf_counter_ns() - depth_started) / 1e6
            reference_roi = fixed_roi & depth_reference
            valid_ratio = float(observed[reference_roi].mean()) if np.any(reference_roi) else 0.0
            occupied = occupancy_from_semantic(semantic)
            obstacles, obstacle_count = bev_obstacle_parameters(occupied, observed, bev)
            mode, status, nmpc_ms, acceleration, omega, minimum_h = solve_shadow(
                solver, path, state, scenario.goal, obstacles
            )
            if valid_ratio < 0.99:
                decision = "abort_perception_invalid"
            elif status != 0:
                decision = "abort_solver_failure"
            elif not all(math.isfinite(value) for value in (acceleration, omega)):
                decision = "abort_nonfinite_command"
            else:
                proposed = integrate_state(state, acceleration, omega)
                decision = supervise_swept_step(grid, state, proposed, scenario.name, time_s)
            target_velocity = float(np.clip(state[3] + acceleration * DT, 0.0, 0.5))
            left_target, right_target = differential_wheel_targets(target_velocity, omega)
            command_applied = decision == "allow"
            targets = np.zeros(len(car.dof_names), dtype=np.float32)
            if command_applied:
                targets[left_index], targets[right_index] = left_target, right_target
            car.set_joint_velocity_targets(targets)
            candidate_pipeline_ms = (time.perf_counter_ns() - pipeline_started) / 1e6
            sensor_to_wheel_ms = (time.perf_counter_ns() - sensor_started) / 1e6
            static_collision = False
            dynamic_collision = False
            for substep in range(1, PHYSICS_STEPS_PER_CONTROL):
                sub_time = time_s + substep * PHYSICS_DT
                sub_dynamic_x, sub_dynamic_y = dynamic_pose(scenario.name, sub_time)
                dynamic.set_world_poses(
                    positions=np.asarray([[sub_dynamic_x, sub_dynamic_y, DYNAMIC_HALF_EXTENT_M]], dtype=np.float32),
                    orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                )
                world.step(render=False)
                actual, _, _, _ = articulation_state(car, left_index, right_index)
                static_collision |= grid.footprint_collision(actual[0], actual[1], actual[2])
                dynamic_collision |= obb_aabb_collision(
                    actual[0], actual[1], actual[2],
                    grid.half_length + grid.margin, grid.half_width + grid.margin,
                    sub_dynamic_x, sub_dynamic_y, DYNAMIC_HALF_EXTENT_M,
                )
            state, z_m, left_measured, right_measured = articulation_state(car, left_index, right_index)
            root_path_length += math.hypot(state[0] - previous_state[0], state[1] - previous_state[1])
            wheel_odometry_distance += 0.5 * (previous_state[3] + state[3]) * DT
            previous_state = state.copy()
            dynamic_distance = math.hypot(state[0] - dynamic_x, state[1] - dynamic_y)
            if scenario.name == "crossing_cart" and dynamic_distance <= 2.2:
                dynamic_encounter_frames += 1
            if scenario.name == "crossing_cart" and dynamic_distance < closest_dynamic:
                closest_dynamic = dynamic_distance
                write_bev_evidence(output / "crossing_cart_closest.png", decoded, occupied, observed, f"Phase5H step={step} distance={dynamic_distance:.2f}m")
            row = {
                "step": step, "time_s": time_s + DT, "x_m": state[0], "y_m": state[1],
                "z_m": z_m, "yaw_rad": state[2], "velocity_mps": state[3],
                "acceleration_mps2": acceleration, "omega_radps": omega,
                "target_velocity_mps": target_velocity, "left_target_radps": left_target,
                "right_target_radps": right_target, "left_measured_radps": left_measured,
                "right_measured_radps": right_measured, "goal_distance_m": goal_distance,
                "goal_yaw_error_rad": goal_yaw_error, "path_progress_m": progress,
                "path_error_m": path.distance_to(state[0], state[1]),
                "candidate_valid_ratio": valid_ratio, "candidate_obstacle_count": obstacle_count,
                "candidate_minimum_h": minimum_h, "solver_status": status,
                "render_ms": render_ms, "inference_ms": inference_ms,
                "depth_lift_ms": depth_lift_ms, "nmpc_ms": nmpc_ms,
                "candidate_pipeline_ms": candidate_pipeline_ms,
                "sensor_to_wheel_ms": sensor_to_wheel_ms, "dynamic_x_m": dynamic_x,
                "dynamic_y_m": dynamic_y, "dynamic_center_distance_m": dynamic_distance,
                "static_collision": int(static_collision), "dynamic_collision": int(dynamic_collision),
                "supervisor_decision": decision, "wheel_command_applied": int(command_applied),
                "physics_feedback": 1, "candidate_controls_articulation": 1,
            }
            writer.writerow(row)
            rows.append(row)
            if not command_applied or static_collision or dynamic_collision:
                abort_reason = decision if not command_applied else "abort_physics_collision"
                car.set_joint_velocity_targets(np.zeros(len(car.dof_names), dtype=np.float32))
                break
            if step % 100 == 99:
                print(f"[Phase 5-H] {scenario.name} step={step + 1} goal={goal_distance:.2f}m v={state[3]:.2f} latency={sensor_to_wheel_ms:.1f}ms")
    car.set_joint_velocity_targets(np.zeros(len(car.dof_names), dtype=np.float32))
    if not rows:
        raise RuntimeError(f"{scenario.name} produced no telemetry")
    terminal_state, _, _, _ = articulation_state(car, left_index, right_index)
    terminal_position_error = math.hypot(scenario.goal[0] - terminal_state[0], scenario.goal[1] - terminal_state[1])
    root_travel = math.hypot(terminal_state[0] - initial_state[0], terminal_state[1] - initial_state[1])
    wheel_odometry_ratio = root_path_length / max(wheel_odometry_distance, 1e-9)
    summary = {
        "name": scenario.name, "steps": len(rows), "duration_s": len(rows) * DT,
        "initial_state": {
            "x_m": float(initial_state[0]), "y_m": float(initial_state[1]),
            "yaw_rad": float(initial_state[2]), "velocity_mps": float(initial_state[3]),
        },
        "reached": reached, "abort_reason": abort_reason,
        "supervisor_aborts": int(abort_reason is not None),
        "solver_failures": sum(row["solver_status"] != 0 for row in rows),
        "static_collision_count": sum(row["static_collision"] for row in rows),
        "dynamic_collision_count": sum(row["dynamic_collision"] for row in rows),
        "wheel_command_applied_ratio": sum(row["wheel_command_applied"] for row in rows) / len(rows),
        "physics_feedback_ratio": sum(row["physics_feedback"] for row in rows) / len(rows),
        "candidate_valid_ratio_mean": float(np.mean([row["candidate_valid_ratio"] for row in rows])),
        "render_p95_ms": percentile(rows, "render_ms"),
        "candidate_pipeline_p95_ms": percentile(rows, "candidate_pipeline_ms"),
        "sensor_to_wheel_p95_ms": percentile(rows, "sensor_to_wheel_ms"),
        "path_error_p95_m": percentile(rows, "path_error_m"),
        "terminal_position_error_m": terminal_position_error,
        "terminal_yaw_error_rad": abs(wrap_angle(scenario.goal[2] - terminal_state[2])),
        "root_travel_m": root_travel, "root_path_length_m": root_path_length,
        "wheel_odometry_distance_m": wheel_odometry_distance,
        "wheel_odometry_ratio": wheel_odometry_ratio,
        "dynamic_encounter_frames": dynamic_encounter_frames,
        "minimum_dynamic_center_distance_m": closest_dynamic if scenario.name == "crossing_cart" else None,
        "telemetry": telemetry_path.name, "telemetry_sha256": sha256(telemetry_path),
    }
    return summary, draw_trajectory(grid, scenario, planned, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", choices=[item.name for item in scenarios()])
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    contract = json.loads((ROOT / "contracts/phase5/phase5h_contract.json").read_text())
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    model_manifest = json.loads(MODEL_MANIFEST.read_text())
    model_path = ROOT / model_manifest["model"]
    if sha256(model_path) != model_manifest["model_sha256"]:
        raise SystemExit("warehouse candidate hash differs from manifest")
    manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    grid = OracleGrid(manifest_path.parent / manifest["archive"], manifest)
    selected = [item for item in scenarios() if args.scenario in (None, item.name)]
    output = args.output or ROOT / "artifacts/phase5h_articulation" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    sensor = scaled_sensor_geometry(phase3["sensor_geometry"])
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 320, "height": 240})
    summaries, evidence = [], []
    try:
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.prims import Articulation, XFormPrim
        from isaacsim.core.utils.stage import open_stage
        from pxr import PhysxSchema, UsdGeom, UsdPhysics

        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        robot_prim = stage.DefinePrim(ROBOT_PATH, "Xform")
        robot_prim.GetReferences().AddReference(str(JETBOT_SOURCE), "/Root/jetbot")
        for prim in stage.Traverse():
            if prim.GetPath().HasPrefix(ROBOT_PATH) and prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(1e5)
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr().Set(100000.0)
        obstacle = UsdGeom.Cube.Define(stage, "/Phase5HDynamicCart")
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        rigid = UsdPhysics.RigidBodyAPI.Apply(obstacle.GetPrim())
        rigid.CreateKinematicEnabledAttr().Set(True)
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
        camera = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA_PATH))
        width, height = sensor["image_size"]
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(sensor["intrinsics"]["fx"] * aperture * height / (sensor["intrinsics"]["fy"] * width))
        camera.CreateFocalLengthAttr(sensor["intrinsics"]["fx"] * aperture / width)
        product = rep.create.render_product(CAMERA_PATH, (width, height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb.attach([product])
        depth.attach([product])
        world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, backend="numpy")
        car = Articulation(prim_paths_expr=ROBOT_PATH, name="phase5h_jetbot")
        dynamic = XFormPrim(prim_paths_expr="/Phase5HDynamicCart", name="phase5h_dynamic_cart")
        world.scene.add(car)
        world.scene.add(dynamic)
        world.reset()
        if car.dof_names != ["left_wheel_joint", "right_wheel_joint"]:
            raise RuntimeError(f"unexpected wheel DOFs: {car.dof_names}")
        world.play()
        model = WarehouseCandidate(model_path)
        for _ in range(3):
            model.infer(np.zeros((240, 320, 3), dtype=np.uint8))
        reset_robot(car, world, selected[0], dynamic)
        for _ in range(contract["latency_optimization"]["warmup_frames_excluded"]):
            world.step(render=True)
            rgb.get_data()
            depth.get_data()
        for scenario in selected:
            summary, image = run_scenario(
                scenario, grid, load_solver(), model, sensor, bev, fixed_roi,
                world, car, dynamic, rgb, depth, output, args.max_steps,
            )
            summaries.append(summary)
            evidence.append(image)
            print(f"{scenario.name}: reached={summary['reached']} abort={summary['abort_reason']} latency_p95={summary['sensor_to_wheel_p95_ms']:.2f}ms")
        rgb.detach([product])
        depth.detach([product])
        world.stop()
    except BaseException:
        app.close()
        raise
    evidence_path = output / "evidence.png"
    if not cv2.imwrite(str(evidence_path), np.concatenate(evidence, axis=1)):
        raise RuntimeError("failed to write Phase 5-H evidence")
    formal = len(selected) == len(contract["scenarios"]) and args.max_steps is None
    acceptance = contract["acceptance"]
    passed = bool(
        formal
        and len(summaries) == acceptance["required_scenarios"]
        and sum(item["reached"] for item in summaries) == acceptance["reached_scenarios"]
        and sum(item["supervisor_aborts"] for item in summaries) == acceptance["supervisor_aborts"]
        and sum(item["static_collision_count"] for item in summaries) == acceptance["static_collision_count"]
        and sum(item["dynamic_collision_count"] for item in summaries) == acceptance["dynamic_collision_count"]
        and sum(item["solver_failures"] for item in summaries) == acceptance["solver_failures"]
        and min(item["wheel_command_applied_ratio"] for item in summaries) >= acceptance["wheel_command_applied_ratio_min"]
        and min(item["physics_feedback_ratio"] for item in summaries) >= acceptance["physics_feedback_ratio_min"]
        and min(item["candidate_valid_ratio_mean"] for item in summaries) >= acceptance["candidate_valid_ratio_mean_min"]
        and max(item["sensor_to_wheel_p95_ms"] for item in summaries) <= acceptance["sensor_to_wheel_p95_ms_max"]
        and max(item["path_error_p95_m"] for item in summaries) <= acceptance["path_error_p95_m_max"]
        and max(item["terminal_position_error_m"] for item in summaries) <= acceptance["terminal_position_error_m_max"]
        and max(item["terminal_yaw_error_rad"] for item in summaries) <= acceptance["terminal_yaw_error_rad_max"]
        and min(item["root_travel_m"] for item in summaries) >= acceptance["minimum_root_travel_m"]
        and min(item["wheel_odometry_ratio"] for item in summaries) >= acceptance["wheel_odometry_ratio_min"]
        and max(item["wheel_odometry_ratio"] for item in summaries) <= acceptance["wheel_odometry_ratio_max"]
        and next(item for item in summaries if item["name"] == "crossing_cart")["dynamic_encounter_frames"] >= acceptance["dynamic_encounter_frames_min"]
    )
    result = {
        "schema_version": "phase5h-articulation-takeover-v1",
        "status": "articulation_gate_passed" if passed else ("articulation_gate_rejected" if formal else "smoke_only"),
        "candidate_controls_articulation": True,
        "state_feedback": "Isaac articulation root pose and measured wheel velocities",
        "direct_pose_updates_after_initialization": 0,
        "oracle_command_override_count": 0,
        "real_vehicle_control_allowed": False,
        "scenarios": summaries,
        "acceptance": acceptance,
        "evidence": evidence_path.name,
        "evidence_sha256": sha256(evidence_path),
        "gate_passed": passed,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))
    print(f"Phase 5-H artifacts: {output}")
    app.close()
    if formal and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
