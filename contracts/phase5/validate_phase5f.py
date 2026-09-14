#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5f_status.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hash(reference, errors, label):
    path = ROOT / reference["path"]
    if not path.is_file():
        errors.append(f"{label} is missing: {path}")
    elif sha256(path) != reference["sha256"]:
        errors.append(f"{label} hash mismatch: {path}")
    return path


def mean(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows)


def percentile(rows, field, quantile):
    values = sorted(float(row[field]) for row in rows)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def close(actual, expected, tolerance=1e-12):
    return abs(actual - expected) <= tolerance


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5f_perception_nmpc_shadow_passed_control_closed":
        errors.append("Phase 5-F frozen status is invalid")
    parent_path = check_hash(status["phase5e_status"], errors, "Phase 5-E status")
    contract_path = check_hash(status["contract"], errors, "Phase 5-F contract")
    check_hash(status["rejected_v1_audit"], errors, "Phase 5-F rejected v1 audit")
    check_hash(status["invalidated_v2_audit"], errors, "Phase 5-F invalidated v2 audit")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    summary_path = check_hash(status["shadow"]["summary"], errors, "Phase 5-F summary")
    telemetry_path = check_hash(status["shadow"]["telemetry"], errors, "Phase 5-F telemetry")
    check_hash(status["shadow"]["evidence"], errors, "Phase 5-F evidence")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with telemetry_path.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    acceptance = contract["acceptance"]
    frame_count = contract["scope"]["frames"]
    if len(rows) != frame_count or summary["frames"] != frame_count:
        errors.append("Phase 5-F evidence must contain exactly 1000 frames")
    if [int(row["source_frame_id"]) for row in rows] != list(range(frame_count)):
        errors.append("Phase 5-F source_frame_id sequence is not exact")
    expected_modes = {
        "center_stop": 400,
        "side_go_left": 100,
        "side_go_right": 100,
        "far_go": 200,
        "absent_go": 200,
    }
    if Counter(row["dynamic_mode"] for row in rows) != expected_modes:
        errors.append("Phase 5-F dynamic schedule drifted")

    eligible_modes = set(contract["metric_definitions"]["control_action_eligible_modes"])
    action_rows = [row for row in rows if row["dynamic_mode"] in eligible_modes]
    if len(action_rows) != contract["metric_definitions"]["control_action_eligible_frames"]:
        errors.append("Phase 5-F control-action eligibility frame count drifted")
    steering_rows = [
        row
        for row in action_rows
        if abs(float(row["oracle_omega_radps"])) >= 0.10
    ]
    oracle_brake = [float(row["oracle_acceleration_mps2"]) <= -0.5 for row in action_rows]
    candidate_brake = [float(row["candidate_acceleration_mps2"]) <= -0.5 for row in action_rows]
    true_brake = sum(a and b for a, b in zip(oracle_brake, candidate_brake))
    missed_brake = sum(a and not b for a, b in zip(oracle_brake, candidate_brake))
    release = [float(row["oracle_acceleration_mps2"]) >= 0.0 for row in action_rows]
    true_release = sum(a and not b for a, b in zip(release, candidate_brake))
    false_brake = sum(a and b for a, b in zip(release, candidate_brake))
    metrics = {
        "control_action_eligible_frames": len(action_rows),
        "acceleration_mae_mps2": mean(action_rows, "acceleration_abs_error_mps2"),
        "omega_mae_radps": mean(action_rows, "omega_abs_error_radps"),
        "steering_direction_agreement": sum(int(row["steering_direction_agree"]) for row in steering_rows)
        / max(len(steering_rows), 1),
        "oracle_solver_success_ratio": sum(int(row["oracle_solver_status"]) == 0 for row in rows) / len(rows),
        "candidate_solver_success_ratio": sum(int(row["candidate_solver_status"]) == 0 for row in rows) / len(rows),
        "runtime_valid_ratio": mean(rows, "candidate_runtime_valid"),
        "candidate_valid_ratio_mean": mean(rows, "candidate_valid_ratio"),
        "oracle_solve_p95_ms": percentile(rows, "oracle_solve_ms", 0.95),
        "candidate_solve_p95_ms": percentile(rows, "candidate_solve_ms", 0.95),
        "candidate_total_p95_ms": percentile(rows, "candidate_total_ms", 0.95),
    }
    brakes = {
        "true_brake": true_brake,
        "missed_brake": missed_brake,
        "true_release": true_release,
        "false_brake": false_brake,
        "oracle_brake_recall": true_brake / max(true_brake + missed_brake, 1),
        "oracle_release_specificity": true_release / max(true_release + false_brake, 1),
    }
    checks = (
        metrics["runtime_valid_ratio"] >= acceptance["runtime_valid_ratio_min"],
        metrics["candidate_valid_ratio_mean"] >= acceptance["candidate_valid_ratio_mean_min"],
        metrics["oracle_solver_success_ratio"] >= acceptance["oracle_solver_success_ratio_min"],
        metrics["candidate_solver_success_ratio"] >= acceptance["candidate_solver_success_ratio_min"],
        metrics["candidate_total_p95_ms"] <= acceptance["candidate_total_latency_p95_ms_max"],
        brakes["oracle_brake_recall"] >= acceptance["oracle_brake_recall_min"],
        brakes["oracle_release_specificity"] >= acceptance["oracle_release_specificity_min"],
        metrics["steering_direction_agreement"] >= acceptance["steering_direction_agreement_min"],
        metrics["acceleration_mae_mps2"] <= acceptance["acceleration_mae_mps2_max"],
        metrics["omega_mae_radps"] <= acceptance["omega_mae_radps_max"],
    )
    if not all(checks):
        errors.append("Phase 5-F independently recomputed metric gate failed")
    for name, value in metrics.items():
        if not close(float(summary["metrics"][name]), value):
            errors.append(f"Phase 5-F summary metric drifted: {name}")
    for name, value in brakes.items():
        if name in summary["brake_decision"] and not close(
            float(summary["brake_decision"][name]), float(value)
        ):
            errors.append(f"Phase 5-F brake metric drifted: {name}")
    if summary["telemetry_sha256"] != sha256(telemetry_path):
        errors.append("Phase 5-F telemetry hash differs from summary")
    if not summary["gate_passed"] or summary["status"] != "shadow_gate_passed":
        errors.append("Phase 5-F summary gate is not frozen as passed")
    if any(int(row["candidate_controls_vehicle"]) for row in rows):
        errors.append("candidate command entered the vehicle control path")
    if summary["control_output_declared"] or summary["candidate_controls_vehicle"]:
        errors.append("Phase 5-F summary grants candidate control authority")
    if not parent["gate"]["live_isaac_shadow_passed"]:
        errors.append("Phase 5-E live Isaac prerequisite is not passed")

    topology_path = ROOT / status["implementation"]["dataflow"]["path"]
    topology = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
    serialized = json.dumps(topology)
    outputs = [output for node in topology["nodes"] for output in node.get("outputs", [])]
    if "control_cmd" in outputs or "control_cmd" in serialized or "fast_brain_nmpc" in serialized:
        errors.append("Phase 5-F topology contains a control output or control consumer")
    shadow_node = next(node for node in topology["nodes"] if node["id"] == "phase5f_dual_nmpc_shadow")
    if shadow_node.get("outputs"):
        errors.append("Phase 5-F dual NMPC shadow node declares an output")
    authority = status["control_authority"]
    if authority["candidate_controls_vehicle"] or authority["control_output_declared"] or authority["control_promotion_allowed"]:
        errors.append("Phase 5-F frozen status opened control authority")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-F perception NMPC shadow validation OK")
    print("1000 exact frames; dual NMPC metrics passed with no candidate control edge")
    print("Oracle NMPC retains sole control authority")


if __name__ == "__main__":
    main()
