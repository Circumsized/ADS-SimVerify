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
from oracle_nmpc_closed_loop import OracleGrid  # noqa: E402
from phase5g_controlled_takeover import dynamic_pose, obb_aabb_collision, scenarios  # noqa: E402
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
from phase5i_runtime import ActuatorCommand, ActuatorWatchdog, dual_resolution_geometry, localization_due  # noqa: E402
from phase5j_faults import CommandGenerationGuard  # noqa: E402
from phase5j_runtime import episode_for_frame  # noqa: E402

ROBOT_PATH = "/Phase5JJetbot"
CAMERA_PATH = f"{ROBOT_PATH}/chassis/rgb_camera/jetbot_camera"
DYNAMIC_PATH = "/Phase5JDynamicCart"


def metadata_value(event, key, default=None):
    metadata = event.get("metadata") or {}
    return metadata.get(key, (metadata.get("parameters") or {}).get(key, default))


def set_wheel_command(car, left_index, right_index, linear, angular):
    left, right = differential_wheel_targets(linear, angular)
    targets = np.zeros(len(car.dof_names), dtype=np.float32)
    targets[left_index], targets[right_index] = left, right
    car.set_joint_velocity_targets(targets)


def watchdog_command(watchdog, now_ms):
    if watchdog.received_ms is None or float(now_ms) - watchdog.received_ms >= watchdog.timeout_ms:
        return ActuatorCommand(0.0, 0.0, "actuator_watchdog_timeout")
    return watchdog.command(now_ms)


def reset_robot(car, world, scenario, dynamic):
    car.set_joint_velocity_targets(np.zeros(len(car.dof_names), dtype=np.float32))
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


def main():
    contract = json.loads((ROOT / "contracts/phase5/phase5j_contract.json").read_text())
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    run_mode = os.environ["PHASE5J_RUN_MODE"]
    generation = int(os.environ["PHASE5J_GENERATION"])
    default_frames = contract["endurance"]["minimum_frames"] if run_mode == "endurance" else contract["fault_gate"]["frames_per_run"]
    max_frames = int(os.environ.get("PHASE5J_MAX_FRAMES", default_frames))
    source_sensor, _ = dual_resolution_geometry(phase3["sensor_geometry"])
    oracle_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_path.read_text())
    grid = OracleGrid(oracle_path.parent / oracle_manifest["archive"], oracle_manifest)
    scenario_by_name = {item.name: item for item in scenarios()}
    scenario_names = tuple(contract["endurance"]["scenarios"])
    frames_per_episode = contract["endurance"]["frames_per_episode"]

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 640, "height": 480})
    try:
        import omni
        import omni.replicator.core as rep
        from dora import Node
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
        obstacle = UsdGeom.Cube.Define(stage, DYNAMIC_PATH)
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        rigid = UsdPhysics.RigidBodyAPI.Apply(obstacle.GetPrim())
        rigid.CreateKinematicEnabledAttr().Set(True)
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
        camera = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA_PATH))
        width, height = source_sensor["image_size"]
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(
            source_sensor["intrinsics"]["fx"] * aperture * height
            / (source_sensor["intrinsics"]["fy"] * width)
        )
        camera.CreateFocalLengthAttr(source_sensor["intrinsics"]["fx"] * aperture / width)
        product = rep.create.render_product(CAMERA_PATH, (width, height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb.attach([product])
        depth.attach([product])
        world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, backend="numpy")
        car = Articulation(prim_paths_expr=ROBOT_PATH, name="phase5j_jetbot")
        dynamic = XFormPrim(prim_paths_expr=DYNAMIC_PATH, name="phase5j_dynamic_cart")
        world.scene.add(car)
        world.scene.add(dynamic)
        world.reset()
        if car.dof_names != ["left_wheel_joint", "right_wheel_joint"]:
            raise RuntimeError(f"unexpected wheel DOFs: {car.dof_names}")
        left_index = car.dof_names.index("left_wheel_joint")
        right_index = car.dof_names.index("right_wheel_joint")
        world.play()
        initial_scenario = scenario_by_name[scenario_names[0]]
        reset_robot(car, world, initial_scenario, dynamic)
        for _ in range(10):
            world.step(render=True)
            rgb.get_data()
            depth.get_data()

        node = Node()
        watchdog = ActuatorWatchdog(timeout_ms=contract["timing"]["actuator_watchdog_ms"])
        command_guard = CommandGenerationGuard(generation)
        freeze_sensor = False
        cached_sensor = None
        current_episode = -1
        sensor_frame_id = -1
        safety_state = "boot"
        safety_reason = "boot"
        reset_applied = False
        last_command_generation = generation
        last_latency_ms = -1.0
        frame_id = 0
        while frame_id < max_frames:
            event = node.next(timeout=0.2)
            now_ms = time.monotonic_ns() / 1e6
            if event is None:
                command = watchdog_command(watchdog, now_ms)
                if command.reason != "fresh":
                    set_wheel_command(car, left_index, right_index, 0.0, 0.0)
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            if event["id"] == "fault_command":
                freeze_sensor = bool(metadata_value(event, "freeze_sensor", False))
                continue
            if event["id"] == "safe_control":
                values = event["value"].to_numpy()
                command_sequence = int(metadata_value(event, "command_sequence_id"))
                command_generation = int(metadata_value(event, "controller_generation"))
                if command_guard.accept(command_generation, command_sequence):
                    watchdog.update(command_sequence, now_ms, float(values[0]), float(values[1]))
                    command = watchdog_command(watchdog, now_ms)
                    set_wheel_command(car, left_index, right_index, command.linear, command.angular)
                    safety_state = str(metadata_value(event, "safety_state", "boot"))
                    safety_reason = str(metadata_value(event, "safety_reason", "none"))
                    reset_applied = bool(metadata_value(event, "reset_applied", False))
                    last_command_generation = command_generation
                    sensor_started_ns = int(metadata_value(event, "sensor_started_ns", -1))
                    proposal_fresh = bool(metadata_value(event, "proposal_fresh", False))
                    last_latency_ms = (
                        (time.perf_counter_ns() - sensor_started_ns) / 1e6
                        if proposal_fresh and sensor_started_ns > 0
                        else -1.0
                    )
                continue
            if event["id"] != "tick":
                continue

            if run_mode == "endurance":
                episode, scenario_name, local_frame = episode_for_frame(frame_id, frames_per_episode, scenario_names)
            else:
                episode, scenario_name, local_frame = 0, "straight_aisle", frame_id
            scenario = scenario_by_name[scenario_name]
            if episode != current_episode:
                reset_robot(car, world, scenario, dynamic)
                watchdog = ActuatorWatchdog(timeout_ms=contract["timing"]["actuator_watchdog_ms"])
                cached_sensor = None
                current_episode = episode

            command = watchdog_command(watchdog, now_ms)
            set_wheel_command(car, left_index, right_index, command.linear, command.angular)
            episode_time_s = local_frame * 0.05
            dynamic_x, dynamic_y = dynamic_pose(scenario_name, episode_time_s)
            dynamic.set_world_poses(
                positions=np.asarray([[dynamic_x, dynamic_y, DYNAMIC_HALF_EXTENT_M]], dtype=np.float32),
                orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
            sensor_started_ns = time.perf_counter_ns()
            world.step(render=True)
            static_collision = False
            dynamic_collision = False
            for substep in range(1, PHYSICS_STEPS_PER_CONTROL):
                sub_time = episode_time_s + substep * PHYSICS_DT
                sub_x, sub_y = dynamic_pose(scenario_name, sub_time)
                dynamic.set_world_poses(
                    positions=np.asarray([[sub_x, sub_y, DYNAMIC_HALF_EXTENT_M]], dtype=np.float32),
                    orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                )
                world.step(render=False)
                actual, _, _, _ = articulation_state(car, left_index, right_index)
                static_collision |= grid.footprint_collision(actual[0], actual[1], actual[2])
                dynamic_collision |= obb_aabb_collision(
                    actual[0], actual[1], actual[2],
                    grid.half_length + grid.margin, grid.half_width + grid.margin,
                    sub_x, sub_y, DYNAMIC_HALF_EXTENT_M,
                )
            state, _, _, _ = articulation_state(car, left_index, right_index)
            full_rgb = np.asarray(rgb.get_data())[:, :, :3].astype(np.uint8)
            full_depth = np.asarray(depth.get_data(), dtype=np.float32)
            full_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
            control_bgr = cv2.resize(full_bgr, (320, 240), interpolation=cv2.INTER_AREA)
            control_depth = cv2.resize(full_depth, (320, 240), interpolation=cv2.INTER_NEAREST)
            if not freeze_sensor or cached_sensor is None:
                sensor_frame_id = frame_id
                sensor_timestamp_ms = time.monotonic_ns() / 1e6
                cached_sensor = (sensor_frame_id, sensor_timestamp_ms, control_bgr.copy(), control_depth.copy(), state.copy())
            packet_frame, packet_timestamp, packet_rgb, packet_depth, packet_state = cached_sensor
            sensor_metadata = {
                "source_frame_id": packet_frame,
                "sensor_frame_id": packet_frame,
                "sensor_timestamp_ms": packet_timestamp,
                "sensor_started_ns": sensor_started_ns,
                "generation": generation,
                "scenario_name": scenario_name,
                "episode_index": episode,
                "episode_time_s": episode_time_s,
            }
            node.send_output("control_rgb", pa.array(packet_rgb.ravel(), type=pa.uint8()), metadata=sensor_metadata)
            node.send_output("metric_depth", pa.array(packet_depth.ravel(), type=pa.float32()), metadata=sensor_metadata)
            node.send_output("vehicle_state", pa.array(packet_state, type=pa.float64()), metadata=sensor_metadata)
            node.send_output(
                "plant_telemetry",
                pa.array(
                    [command.linear, command.angular, state[0], state[1], state[2], state[3],
                     last_latency_ms, int(static_collision), int(dynamic_collision)],
                    type=pa.float32(),
                ),
                metadata={
                    "source_frame_id": frame_id,
                    "run_mode": run_mode,
                    "scenario_name": scenario_name,
                    "episode_index": episode,
                    "sensor_frame_id": packet_frame,
                    "sensor_frozen": freeze_sensor,
                    "controller_generation": last_command_generation,
                    "safety_state": safety_state,
                    "safety_reason": safety_reason,
                    "reset_applied": reset_applied,
                    "actuator_reason": command.reason,
                },
            )
            reset_applied = False
            if localization_due(frame_id):
                ok, encoded = cv2.imencode(".jpg", full_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    node.send_output(
                        "localization_image_640",
                        pa.array(encoded.ravel(), type=pa.uint8()),
                        metadata={"source_frame_id": frame_id, "shape": [480, 640, 3]},
                    )
            if frame_id % 100 == 99:
                print(
                    f"[Phase 5-J {run_mode}] {frame_id + 1}/{max_frames} "
                    f"scenario={scenario_name} safety={safety_state} actuator={command.reason}"
                )
            frame_id += 1

        set_wheel_command(car, left_index, right_index, 0.0, 0.0)
        node.send_output(
            "run_complete",
            pa.array([frame_id], type=pa.int64()),
            metadata={"frames": frame_id, "run_mode": run_mode, "generation": generation},
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
