#!/usr/bin/env python3
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contracts/phase5"))
from oracle_nmpc_closed_loop import OracleGrid, Polyline, astar, load_solver, prune_path  # noqa: E402
from phase5b_shadow_replay import control_roi, occupancy_from_semantic  # noqa: E402
from phase5c3_candidate_shadow import WarehouseCandidate, candidate_depth_lift  # noqa: E402
from phase5f_dual_nmpc_shadow import bev_obstacle_parameters, solve_shadow  # noqa: E402
from phase5g_controlled_takeover import integrate_state, scenarios, supervise_swept_step  # noqa: E402
from phase5h_articulation_takeover import (  # noqa: E402
    DYNAMIC_HALF_EXTENT_M,
    JETBOT_SOURCE,
    PHASE4_OVERLAY,
    PHYSICS_DT,
    PHYSICS_STEPS_PER_CONTROL,
    articulation_state,
    differential_wheel_targets,
    quaternion_from_yaw,
)
from phase5i_runtime import (  # noqa: E402
    ActuatorWatchdog,
    dual_resolution_geometry,
    localization_due,
    proposal_timestamp_ms,
)

ROBOT_PATH = "/Phase5IJetbot"
CAMERA_PATH = f"{ROBOT_PATH}/chassis/rgb_camera/jetbot_camera"
MODEL_MANIFEST = ROOT / "model/warehouse_nav14_candidate.json"


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_metadata(event, key):
    metadata = event.get("metadata") or {}
    if key in metadata:
        return metadata[key]
    return (metadata.get("parameters") or {}).get(key)


def set_wheel_command(car, left_index, right_index, linear, angular):
    left, right = differential_wheel_targets(linear, angular)
    targets = np.zeros(len(car.dof_names), dtype=np.float32)
    targets[left_index], targets[right_index] = left, right
    car.set_joint_velocity_targets(targets)


def main():
    contract = json.loads((ROOT / "contracts/phase5/phase5i_contract.json").read_text())
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    run_mode = os.environ.get("PHASE5I_RUN_MODE", "nominal")
    if run_mode not in set(contract["multirun_gate"]["runs"]):
        raise SystemExit(f"unsupported PHASE5I_RUN_MODE: {run_mode}")
    max_frames = int(os.environ.get("PHASE5I_MAX_FRAMES", contract["multirun_gate"]["frames_per_run"]))
    sensor, control_sensor = dual_resolution_geometry(phase3["sensor_geometry"])
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    oracle_manifest = json.loads(manifest_path.read_text())
    grid = OracleGrid(manifest_path.parent / oracle_manifest["archive"], oracle_manifest)
    scenario = scenarios()[0]
    path = Polyline(prune_path(grid, astar(grid, scenario.start[:2], scenario.goal[:2])))
    model_manifest = json.loads(MODEL_MANIFEST.read_text())
    model_path = ROOT / model_manifest["model"]
    if sha256(model_path) != model_manifest["model_sha256"]:
        raise SystemExit("warehouse candidate hash differs from manifest")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 640, "height": 480})
    try:
        import omni
        import omni.replicator.core as rep
        from dora import Node
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
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
        car = Articulation(prim_paths_expr=ROBOT_PATH, name="phase5i_jetbot")
        world.scene.add(car)
        world.reset()
        if car.dof_names != ["left_wheel_joint", "right_wheel_joint"]:
            raise RuntimeError(f"unexpected wheel DOFs: {car.dof_names}")
        left_index = car.dof_names.index("left_wheel_joint")
        right_index = car.dof_names.index("right_wheel_joint")
        car.set_world_poses(
            np.asarray([[scenario.start[0], scenario.start[1], 0.05]], dtype=np.float32),
            quaternion_from_yaw(scenario.start[2]),
        )
        car.set_joint_velocities(np.zeros((1, len(car.dof_names)), dtype=np.float32))
        car.set_velocities(np.zeros((1, 6), dtype=np.float32))
        world.play()
        for _ in range(20):
            world.step(render=False)
        model = WarehouseCandidate(model_path)
        for _ in range(3):
            model.infer(np.zeros((240, 320, 3), dtype=np.uint8))
        for _ in range(10):
            world.step(render=True)
            rgb.get_data()
            depth.get_data()
        solver = load_solver()
        watchdog = ActuatorWatchdog(timeout_ms=contract["timing"]["actuator_watchdog_ms"])
        node = Node()
        pending = None
        frame_id = 0
        while frame_id < max_frames:
            event = node.next(timeout=0.01)
            now_ms = time.monotonic_ns() / 1e6
            if event is None:
                if pending is not None and now_ms - pending["proposal_sent_ms"] > watchdog.timeout_ms:
                    set_wheel_command(car, left_index, right_index, 0.0, 0.0)
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            if event["id"] == "tick" and pending is None:
                sensor_started_ns = time.perf_counter_ns()
                render_started = time.perf_counter_ns()
                world.step(render=True)
                render_ms = (time.perf_counter_ns() - render_started) / 1e6
                state, _, _, _ = articulation_state(car, left_index, right_index)
                rgb_data = np.asarray(rgb.get_data())
                depth_data = np.asarray(depth.get_data(), dtype=np.float32)
                sensor_valid = rgb_data.shape[:2] == (480, 640) and depth_data.shape == (480, 640)
                if sensor_valid:
                    full_bgr = cv2.cvtColor(rgb_data[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
                    control_bgr = cv2.resize(full_bgr, (320, 240), interpolation=cv2.INTER_AREA)
                    control_depth_data = cv2.resize(depth_data, (320, 240), interpolation=cv2.INTER_NEAREST)
                    classes, inference_ms = model.infer(control_bgr)
                    lift_started = time.perf_counter_ns()
                    semantic, observed = candidate_depth_lift(classes, control_depth_data, control_sensor, bev)
                    dummy = np.full(control_depth_data.shape, 13, dtype=np.uint8)
                    _, depth_reference = candidate_depth_lift(dummy, control_depth_data, control_sensor, bev)
                    depth_lift_ms = (time.perf_counter_ns() - lift_started) / 1e6
                    reference_roi = fixed_roi & depth_reference
                    valid_ratio = float(observed[reference_roi].mean()) if np.any(reference_roi) else 0.0
                    occupied = occupancy_from_semantic(semantic)
                    obstacles, _ = bev_obstacle_parameters(occupied, observed, bev)
                    _, solver_status, nmpc_ms, acceleration, omega, _ = solve_shadow(
                        solver, path, state, scenario.goal, obstacles
                    )
                    perception_valid = valid_ratio >= 0.99
                    solver_valid = solver_status == 0 and all(math.isfinite(value) for value in (acceleration, omega))
                    if solver_valid:
                        decision = supervise_swept_step(
                            grid, state, integrate_state(state, acceleration, omega), scenario.name, frame_id * 0.05
                        )
                        solver_valid &= decision == "allow"
                    target_velocity = float(np.clip(state[3] + acceleration * 0.05, 0.0, 0.5)) if solver_valid else 0.0
                else:
                    full_bgr = None
                    inference_ms = depth_lift_ms = nmpc_ms = 0.0
                    perception_valid = solver_valid = False
                    target_velocity = omega = 0.0
                health_timestamp_ms = time.monotonic_ns() / 1e6
                proposal_timestamp = proposal_timestamp_ms(
                    run_mode, frame_id, health_timestamp_ms, contract["timing"]["supervisor_watchdog_ms"]
                )
                metadata = {
                    "source_frame_id": frame_id,
                    "health_timestamp_ms": health_timestamp_ms,
                    "proposal_timestamp_ms": proposal_timestamp,
                    "sensor_valid": sensor_valid,
                    "perception_valid": perception_valid,
                    "solver_valid": solver_valid,
                    "articulation_ready": True,
                    "source_resolution": [640, 480],
                    "control_resolution": [320, 240],
                    "run_mode": run_mode,
                }
                node.send_output(
                    "proposed_control",
                    pa.array([target_velocity, float(omega)], type=pa.float32()),
                    metadata=metadata,
                )
                node.send_output(
                    "runtime_health",
                    pa.array([frame_id, int(sensor_valid), int(perception_valid), int(solver_valid)], type=pa.float32()),
                    metadata=metadata,
                )
                pending = {
                    "frame_id": frame_id,
                    "sensor_started_ns": sensor_started_ns,
                    "proposal_sent_ms": time.monotonic_ns() / 1e6,
                    "render_ms": render_ms,
                    "inference_ms": inference_ms,
                    "depth_lift_ms": depth_lift_ms,
                    "nmpc_ms": nmpc_ms,
                    "full_bgr": full_bgr,
                }
                continue
            if event["id"] != "safe_control" or pending is None:
                continue
            safe_frame_id = int(event_metadata(event, "source_frame_id"))
            if safe_frame_id != pending["frame_id"]:
                continue
            safe = event["value"].to_numpy()
            accepted = watchdog.update(safe_frame_id, now_ms, float(safe[0]), float(safe[1]))
            command = watchdog.command(now_ms) if accepted else watchdog.command(now_ms + watchdog.timeout_ms + 1.0)
            set_wheel_command(car, left_index, right_index, command.linear, command.angular)
            sensor_to_wheel_ms = (time.perf_counter_ns() - pending["sensor_started_ns"]) / 1e6
            for _ in range(1, PHYSICS_STEPS_PER_CONTROL):
                world.step(render=False)
            state, _, _, _ = articulation_state(car, left_index, right_index)
            static_collision = grid.footprint_collision(state[0], state[1], state[2])
            metadata = {
                "source_frame_id": safe_frame_id,
                "run_mode": run_mode,
                "safety_state": str(event_metadata(event, "safety_state")),
                "safety_reason": str(event_metadata(event, "safety_reason")),
                "reset_applied": bool(event_metadata(event, "reset_applied")),
                "actuator_reason": command.reason,
            }
            node.send_output(
                "articulation_telemetry",
                pa.array(
                    [
                        command.linear, command.angular, state[0], state[1], state[2], state[3],
                        pending["render_ms"], pending["inference_ms"], pending["depth_lift_ms"],
                        pending["nmpc_ms"], sensor_to_wheel_ms, int(static_collision), 0,
                    ],
                    type=pa.float32(),
                ),
                metadata=metadata,
            )
            if localization_due(safe_frame_id) and pending["full_bgr"] is not None:
                ok, encoded = cv2.imencode(
                    ".jpg", pending["full_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                if ok:
                    node.send_output(
                        "localization_image_640",
                        pa.array(encoded.ravel(), type=pa.uint8()),
                        metadata={
                            "source_frame_id": safe_frame_id,
                            "shape": [480, 640, 3],
                            "consumer": "XFeat localization only",
                            "control_critical_path": False,
                        },
                    )
            if frame_id % 20 == 19:
                print(
                    f"[Phase 5-I {run_mode}] frames={frame_id + 1}/{max_frames} "
                    f"state={metadata['safety_state']} latency={sensor_to_wheel_ms:.2f}ms"
                )
            pending = None
            frame_id += 1
        set_wheel_command(car, left_index, right_index, 0.0, 0.0)
        node.send_output(
            "run_complete",
            pa.array([frame_id], type=pa.int64()),
            metadata={"frames": frame_id, "run_mode": run_mode},
        )
        rgb.detach([product])
        depth.detach([product])
        world.stop()
    except BaseException:
        app.close()
        raise
    app.close()


if __name__ == "__main__":
    main()
