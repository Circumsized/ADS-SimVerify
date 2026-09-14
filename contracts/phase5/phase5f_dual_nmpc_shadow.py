#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from dora import Node

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contracts/phase5"))
from oracle_nmpc_closed_loop import (  # noqa: E402
    OracleGrid,
    Polyline,
    Scenario,
    astar,
    load_solver,
    minimum_predicted_obstacle_h,
    prune_path,
    set_nmpc_problem,
    target_mode,
)
from phase5b_shadow_replay import control_roi, load_trajectory  # noqa: E402
from phase5d_runtime_node import metadata_value, source_frame_id  # noqa: E402

DUMMY_OBSTACLE = (1000.0, 1000.0, 0.10, 0.10)
BRAKE_THRESHOLD_MPS2 = -0.50
CSV_FIELDS = [
    "source_frame_id", "scenario", "scenario_step", "dynamic_mode",
    "x_m", "y_m", "yaw_rad", "velocity_mps", "target_mode",
    "candidate_runtime_valid", "oracle_valid_ratio", "candidate_valid_ratio",
    "oracle_obstacle_count", "candidate_obstacle_count",
    "oracle_center_obstacle_m", "candidate_center_obstacle_m",
    "oracle_solver_status", "candidate_solver_status",
    "oracle_solve_ms", "candidate_solve_ms", "candidate_perception_ms",
    "candidate_total_ms", "oracle_acceleration_mps2", "candidate_acceleration_mps2",
    "oracle_omega_radps", "candidate_omega_radps", "acceleration_abs_error_mps2",
    "omega_abs_error_radps", "oracle_minimum_h", "candidate_minimum_h",
    "oracle_brake", "candidate_brake", "steering_direction_agree",
    "candidate_controls_vehicle",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile * 100.0))


def bev_obstacle_parameters(occupied, valid, bev):
    occupied = np.asarray(occupied, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    expected_shape = tuple(bev["shape"][:2])
    if occupied.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(f"BEV shape mismatch: expected {expected_shape}")
    rows, cols = np.nonzero(occupied & valid)
    forward = (bev["ego_origin_cell"][0] - rows) * bev["meters_per_cell"]
    left = (bev["ego_origin_cell"][1] - cols) * bev["meters_per_cell"]
    roi = (forward >= 0.1) & (forward <= 2.2) & (np.abs(left) <= 0.8)
    forward, left = forward[roi], left[roi]
    sectors = (left > 0.15, np.abs(left) <= 0.15, left < -0.15)
    parameters = []
    count = 0
    for sector in sectors:
        indices = np.flatnonzero(sector)
        if indices.size:
            index = indices[int(np.argmin(forward[indices]))]
            parameters.extend((float(forward[index]), float(left[index]), 0.35, 0.25))
            count += 1
        else:
            parameters.extend(DUMMY_OBSTACLE)
    return np.asarray(parameters, dtype=np.float64), count


def center_obstacle_distance(parameters):
    distance = float(parameters[4])
    return None if distance >= 999.0 else distance


def brake_metrics(rows):
    brake_rows = [row for row in rows if row["oracle_acceleration_mps2"] <= BRAKE_THRESHOLD_MPS2]
    release_rows = [row for row in rows if row["oracle_acceleration_mps2"] >= 0.0]
    true_brake = sum(row["candidate_brake"] for row in brake_rows)
    missed_brake = len(brake_rows) - true_brake
    false_brake = sum(row["candidate_brake"] for row in release_rows)
    true_release = len(release_rows) - false_brake
    center_rows = [row for row in rows if row["dynamic_mode"] == "center_stop"]
    return {
        "true_brake": true_brake,
        "missed_brake": missed_brake,
        "true_release": true_release,
        "false_brake": false_brake,
        "moderate_oracle_deceleration_excluded": len(rows) - len(brake_rows) - len(release_rows),
        "oracle_brake_recall": true_brake / max(true_brake + missed_brake, 1),
        "oracle_release_specificity": true_release / max(true_release + false_brake, 1),
        "center_stop_brake_recall": sum(row["candidate_brake"] for row in center_rows)
        / max(len(center_rows), 1),
    }


def scenario_paths(grid):
    scenarios = (
        Scenario("straight_aisle", (1.0, -2.0, 0.0, 0.0), (7.0, -2.0, 0.0)),
        Scenario("diagonal_turn", (1.0, -2.0, math.pi / 2.0, 0.0), (7.0, 2.0, 0.0)),
        Scenario("pallet_detour", (-4.0, 13.5, 0.0, 0.0), (6.5, 13.5, 0.0)),
    )
    return {
        scenario.name: (
            scenario,
            Polyline(prune_path(grid, astar(grid, scenario.start[:2], scenario.goal[:2]))),
        )
        for scenario in scenarios
    }


def solve_shadow(solver, path, state, goal, obstacles):
    progress = path.nearest_s(state[0], state[1])
    mode, speed, heading_error = target_mode(path, state, progress, goal)
    set_nmpc_problem(
        solver, path, state, progress, goal, speed, mode, heading_error, obstacles
    )
    started = time.perf_counter_ns()
    status = int(solver.solve())
    solve_ms = (time.perf_counter_ns() - started) / 1e6
    if status == 0:
        acceleration, omega = (float(value) for value in solver.get(0, "u"))
        minimum_h = minimum_predicted_obstacle_h(solver, obstacles)
    else:
        acceleration, omega, minimum_h = -1.0, 0.0, 0.0
    return mode, status, solve_ms, float(acceleration), float(omega), float(minimum_h)


def draw_plot(rows, output):
    canvas = np.full((640, 1280, 3), (248, 248, 248), dtype=np.uint8)
    cv2.putText(canvas, "Phase 5-F dual NMPC shadow", (28, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)

    def plot(rect, field_a, field_b, low, high, title):
        x0, y0, x1, y1 = rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 40), 1)
        cv2.putText(canvas, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        for field, color in ((field_a, (30, 90, 220)), (field_b, (30, 165, 70))):
            points = []
            for index, row in enumerate(rows):
                x = x0 + int(index * (x1 - x0) / max(len(rows) - 1, 1))
                value = max(low, min(high, float(row[field])))
                y = y1 - int((value - low) * (y1 - y0) / (high - low))
                points.append((x, y))
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)

    plot((45, 90, 1235, 315), "oracle_acceleration_mps2", "candidate_acceleration_mps2", -1.0, 1.0, "acceleration: Oracle=red candidate=green")
    plot((45, 380, 1235, 605), "oracle_omega_radps", "candidate_omega_radps", -0.6, 0.6, "yaw rate: Oracle=red candidate=green")
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("failed to write Phase 5-F evidence")


def summarize(rows, contract, telemetry_path, evidence_path):
    acceptance = contract["acceptance"]
    eligible_modes = set(contract["metric_definitions"]["control_action_eligible_modes"])
    action_rows = [row for row in rows if row["dynamic_mode"] in eligible_modes]
    brakes = brake_metrics(action_rows)
    steering_rows = [
        row
        for row in action_rows
        if abs(row["oracle_omega_radps"]) >= 0.10
    ]
    steering_agreement = sum(row["steering_direction_agree"] for row in steering_rows) / max(len(steering_rows), 1)
    metrics = {
        "control_action_eligible_frames": len(action_rows),
        "acceleration_mae_mps2": float(np.mean([row["acceleration_abs_error_mps2"] for row in action_rows])),
        "omega_mae_radps": float(np.mean([row["omega_abs_error_radps"] for row in action_rows])),
        "steering_direction_agreement": steering_agreement,
        "oracle_solver_success_ratio": sum(row["oracle_solver_status"] == 0 for row in rows) / len(rows),
        "candidate_solver_success_ratio": sum(row["candidate_solver_status"] == 0 for row in rows) / len(rows),
        "runtime_valid_ratio": sum(row["candidate_runtime_valid"] for row in rows) / len(rows),
        "candidate_valid_ratio_mean": float(np.mean([row["candidate_valid_ratio"] for row in rows])),
        "oracle_solve_p95_ms": percentile([row["oracle_solve_ms"] for row in rows], 0.95),
        "candidate_solve_p95_ms": percentile([row["candidate_solve_ms"] for row in rows], 0.95),
        "candidate_total_p95_ms": percentile([row["candidate_total_ms"] for row in rows], 0.95),
        "candidate_minimum_predicted_h": min(row["candidate_minimum_h"] for row in rows),
    }
    exact_ids = [row["source_frame_id"] for row in rows] == list(range(len(rows)))
    passed = bool(
        len(rows) == contract["scope"]["frames"]
        and exact_ids
        and metrics["runtime_valid_ratio"] >= acceptance["runtime_valid_ratio_min"]
        and metrics["candidate_valid_ratio_mean"] >= acceptance["candidate_valid_ratio_mean_min"]
        and metrics["oracle_solver_success_ratio"] >= acceptance["oracle_solver_success_ratio_min"]
        and metrics["candidate_solver_success_ratio"] >= acceptance["candidate_solver_success_ratio_min"]
        and metrics["candidate_total_p95_ms"] <= acceptance["candidate_total_latency_p95_ms_max"]
        and brakes["oracle_brake_recall"] >= acceptance["oracle_brake_recall_min"]
        and brakes["oracle_release_specificity"] >= acceptance["oracle_release_specificity_min"]
        and metrics["steering_direction_agreement"] >= acceptance["steering_direction_agreement_min"]
        and metrics["acceleration_mae_mps2"] <= acceptance["acceleration_mae_mps2_max"]
        and metrics["omega_mae_radps"] <= acceptance["omega_mae_radps_max"]
    )
    return {
        "schema_version": "phase5f-dual-nmpc-shadow-v3",
        "status": "shadow_gate_passed" if passed else "shadow_gate_rejected",
        "frames": len(rows),
        "source_frame_exact": exact_ids,
        "dynamic_modes": dict(Counter(row["dynamic_mode"] for row in rows)),
        "comparison_design": {
            "state_and_reference_path_identical": True,
            "oracle_local_obstacles": "Phase 5-E live USD Oracle BEV",
            "candidate_local_obstacles": "warehouse_nav14 candidate metric-depth BEV",
            "candidate_command_applied": False,
        },
        "metrics": metrics,
        "brake_decision": brakes,
        "acceptance": acceptance,
        "telemetry": telemetry_path.name,
        "telemetry_sha256": sha256(telemetry_path),
        "evidence": evidence_path.name,
        "evidence_sha256": sha256(evidence_path),
        "control_output_declared": False,
        "candidate_controls_vehicle": False,
        "control_promotion_allowed": False,
        "gate_passed": passed,
    }


def main():
    output = Path(os.environ["PHASE5F_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((ROOT / "contracts/phase5/phase5f_contract.json").read_text())
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    frame_count = int(os.environ.get("PHASE5F_MAX_FRAMES", contract["scope"]["frames"]))
    if not 1 <= frame_count <= contract["scope"]["frames"]:
        raise SystemExit("PHASE5F_MAX_FRAMES is outside the frozen range")
    _, trajectory = load_trajectory(phase5a)
    trajectory = trajectory[:frame_count]
    manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    grid = OracleGrid(manifest_path.parent / manifest["archive"], manifest)
    paths = scenario_paths(grid)
    oracle_solver, candidate_solver = load_solver(), load_solver()
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    streams = {
        name: {}
        for name in (
            "oracle_bev",
            "oracle_valid",
            "depth_reference_valid",
            "shadow_bev_grid",
            "shadow_valid",
        )
    }
    oracle_metadata, candidate_metadata = {}, {}
    rows = []
    node = Node()
    telemetry_path = output / "frames.csv"
    with telemetry_path.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        while len(rows) < frame_count:
            event = node.next(timeout=2.0)
            if event is None:
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] not in streams:
                continue
            frame_id = source_frame_id(event)
            if frame_id is None or frame_id >= frame_count:
                continue
            stream = event["id"]
            streams[stream][frame_id] = event["value"].to_numpy().copy()
            if stream == "oracle_bev":
                oracle_metadata[frame_id] = str(metadata_value(event, "dynamic_mode"))
            elif stream == "shadow_bev_grid":
                candidate_metadata[frame_id] = (
                    bool(metadata_value(event, "valid")),
                    float(metadata_value(event, "latency_ms") or 0.0),
                )
            if not all(frame_id in values for values in streams.values()):
                continue
            frame = trajectory[frame_id]
            scenario, path = paths[frame["scenario"]]
            state = np.asarray(
                [frame["x_m"], frame["y_m"], frame["yaw_rad"], frame["velocity_mps"]],
                dtype=np.float64,
            )
            oracle = streams["oracle_bev"].pop(frame_id).reshape(192, 192) > 0
            oracle_valid = streams["oracle_valid"].pop(frame_id).reshape(192, 192) > 0
            depth_reference_valid = streams["depth_reference_valid"].pop(frame_id).reshape(192, 192) > 0
            candidate = streams["shadow_bev_grid"].pop(frame_id).reshape(192, 192) > 0
            candidate_valid = streams["shadow_valid"].pop(frame_id).reshape(192, 192) > 0
            oracle_parameters, oracle_count = bev_obstacle_parameters(oracle, oracle_valid, bev)
            candidate_parameters, candidate_count = bev_obstacle_parameters(candidate, candidate_valid, bev)
            oracle_result = solve_shadow(oracle_solver, path, state, scenario.goal, oracle_parameters)
            candidate_result = solve_shadow(candidate_solver, path, state, scenario.goal, candidate_parameters)
            mode, oracle_status, oracle_ms, oracle_a, oracle_w, oracle_h = oracle_result
            _, candidate_status, candidate_ms, candidate_a, candidate_w, candidate_h = candidate_result
            runtime_valid, perception_ms = candidate_metadata.pop(frame_id)
            oracle_brake = oracle_a <= BRAKE_THRESHOLD_MPS2
            candidate_brake = candidate_a <= BRAKE_THRESHOLD_MPS2
            steering_agree = abs(oracle_w) < 0.05 or oracle_w * candidate_w > 0.0
            row = {
                "source_frame_id": frame_id,
                "scenario": frame["scenario"],
                "scenario_step": frame["scenario_step"],
                "dynamic_mode": oracle_metadata.pop(frame_id),
                "x_m": frame["x_m"], "y_m": frame["y_m"], "yaw_rad": frame["yaw_rad"],
                "velocity_mps": frame["velocity_mps"], "target_mode": mode,
                "candidate_runtime_valid": int(runtime_valid),
                "oracle_valid_ratio": float(oracle_valid[fixed_roi].mean()),
                "candidate_valid_ratio": float(
                    candidate_valid[fixed_roi & oracle_valid & depth_reference_valid].mean()
                ),
                "oracle_obstacle_count": oracle_count, "candidate_obstacle_count": candidate_count,
                "oracle_center_obstacle_m": center_obstacle_distance(oracle_parameters) or "",
                "candidate_center_obstacle_m": center_obstacle_distance(candidate_parameters) or "",
                "oracle_solver_status": oracle_status, "candidate_solver_status": candidate_status,
                "oracle_solve_ms": oracle_ms, "candidate_solve_ms": candidate_ms,
                "candidate_perception_ms": perception_ms,
                "candidate_total_ms": perception_ms + candidate_ms,
                "oracle_acceleration_mps2": oracle_a, "candidate_acceleration_mps2": candidate_a,
                "oracle_omega_radps": oracle_w, "candidate_omega_radps": candidate_w,
                "acceleration_abs_error_mps2": abs(oracle_a - candidate_a),
                "omega_abs_error_radps": abs(oracle_w - candidate_w),
                "oracle_minimum_h": oracle_h, "candidate_minimum_h": candidate_h,
                "oracle_brake": int(oracle_brake), "candidate_brake": int(candidate_brake),
                "steering_direction_agree": int(steering_agree),
                "candidate_controls_vehicle": 0,
            }
            writer.writerow(row)
            target.flush()
            rows.append(row)
            if len(rows) % 100 == 0:
                print(
                    f"[Phase 5-F dual NMPC] frames={len(rows)}/{frame_count} "
                    f"oracle=({oracle_a:+.2f},{oracle_w:+.2f}) "
                    f"shadow=({candidate_a:+.2f},{candidate_w:+.2f})"
                )
    evidence_path = output / "evidence.png"
    draw_plot(rows, evidence_path)
    summary = summarize(rows, contract, telemetry_path, evidence_path)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2))
    print(f"Phase 5-F artifacts: {output}")
    if frame_count == contract["scope"]["frames"] and not summary["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
