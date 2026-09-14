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
from phase5i_runtime import dual_resolution_geometry  # noqa: E402
from phase5j_faults import SensorSequenceGuard  # noqa: E402
from phase5j_runtime import FrameTripletBuffer, transport_safe_clip  # noqa: E402

MODEL_MANIFEST = ROOT / "model/warehouse_nav14_candidate.json"


def sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_value(event, key):
    metadata = event.get("metadata") or {}
    return metadata.get(key, (metadata.get("parameters") or {}).get(key))


def main():
    generation = int(os.environ["PHASE5J_GENERATION"])
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    _, sensor = dual_resolution_geometry(phase3["sensor_geometry"])
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    oracle_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_path.read_text())
    grid = OracleGrid(oracle_path.parent / oracle_manifest["archive"], oracle_manifest)
    scenario_by_name = {item.name: item for item in scenarios()}
    paths = {
        name: Polyline(prune_path(grid, astar(grid, item.start[:2], item.goal[:2])))
        for name, item in scenario_by_name.items()
    }
    model_manifest = json.loads(MODEL_MANIFEST.read_text())
    model_path = ROOT / model_manifest["model"]
    if sha256(model_path) != model_manifest["model_sha256"]:
        raise SystemExit("warehouse candidate hash differs from manifest")
    model = WarehouseCandidate(model_path)
    for _ in range(3):
        model.infer(np.zeros((240, 320, 3), dtype=np.uint8))
    solver = load_solver()
    guard = SensorSequenceGuard(generation)
    pairer = FrameTripletBuffer(max_pending=4)

    from dora import Node

    node = Node()
    pid = os.getpid()
    node.send_output(
        "controller_heartbeat",
        pa.array([pid, generation], type=pa.int64()),
        metadata={"pid": pid, "generation": generation, "status": "ready"},
    )
    while True:
        event = node.next(timeout=1.0)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "run_complete":
            break
        kind_by_input = {"control_rgb": "rgb", "metric_depth": "depth", "vehicle_state": "state"}
        kind = kind_by_input.get(event["id"])
        if kind is None:
            continue
        frame_id = int(metadata_value(event, "sensor_frame_id"))
        packet = pairer.add(kind, frame_id, event)
        if packet is None:
            continue
        metadata_event = packet["state"]
        timestamp_ms = float(metadata_value(metadata_event, "sensor_timestamp_ms"))
        sensor_generation = int(metadata_value(metadata_event, "generation"))
        valid_sequence, sequence_reason = guard.observe(sensor_generation, frame_id, timestamp_ms)
        if not valid_sequence:
            node.send_output(
                "controller_heartbeat",
                pa.array([pid, generation], type=pa.int64()),
                metadata={"pid": pid, "generation": generation, "status": sequence_reason},
            )
            continue
        scenario_name = str(metadata_value(metadata_event, "scenario_name"))
        scenario = scenario_by_name[scenario_name]
        state = packet["state"]["value"].to_numpy().astype(np.float64, copy=False)
        rgb = packet["rgb"]["value"].to_numpy().astype(np.uint8, copy=False).reshape(240, 320, 3)
        depth = packet["depth"]["value"].to_numpy().astype(np.float32, copy=False).reshape(240, 320)
        classes, _ = model.infer(rgb)
        semantic, observed = candidate_depth_lift(classes, depth, sensor, bev)
        dummy = np.full(depth.shape, 13, dtype=np.uint8)
        _, depth_reference = candidate_depth_lift(dummy, depth, sensor, bev)
        reference_roi = fixed_roi & depth_reference
        valid_ratio = float(observed[reference_roi].mean()) if np.any(reference_roi) else 0.0
        occupied = occupancy_from_semantic(semantic)
        obstacles, _ = bev_obstacle_parameters(occupied, observed, bev)
        _, solver_status, _, acceleration, omega, _ = solve_shadow(
            solver, paths[scenario_name], state, scenario.goal, obstacles
        )
        perception_valid = valid_ratio >= 0.99
        solver_valid = solver_status == 0 and all(math.isfinite(value) for value in (acceleration, omega))
        if solver_valid:
            decision = supervise_swept_step(
                grid,
                state,
                integrate_state(state, acceleration, omega),
                scenario_name,
                float(metadata_value(metadata_event, "episode_time_s")),
            )
            solver_valid = decision == "allow"
        target_velocity = float(np.clip(state[3] + acceleration * 0.05, 0.0, 0.5)) if solver_valid else 0.0
        omega = transport_safe_clip(omega, 0.6)
        now_ms = time.monotonic_ns() / 1e6
        metadata = {
            "source_frame_id": frame_id,
            "health_timestamp_ms": now_ms,
            "proposal_timestamp_ms": now_ms,
            "sensor_valid": True,
            "perception_valid": perception_valid,
            "solver_valid": solver_valid,
            "articulation_ready": True,
            "controller_generation": generation,
            "sensor_started_ns": int(metadata_value(metadata_event, "sensor_started_ns")),
            "scenario_name": scenario_name,
        }
        node.send_output(
            "proposed_control",
            pa.array([target_velocity, float(omega)], type=pa.float32()),
            metadata=metadata,
        )
        node.send_output(
            "controller_heartbeat",
            pa.array([pid, generation], type=pa.int64()),
            metadata={"pid": pid, "generation": generation, "status": "active", "source_frame_id": frame_id},
        )


if __name__ == "__main__":
    main()
