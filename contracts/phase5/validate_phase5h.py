#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5h_status.json"
DT = 0.05
WHEEL_RADIUS_M = 0.03362
WHEEL_BASE_M = 0.1125


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


def percentile(rows, field, quantile=0.95):
    values = sorted(float(row[field]) for row in rows)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def close(actual, expected, tolerance=1e-9):
    return abs(float(actual) - float(expected)) <= tolerance


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5h_isaac_articulation_passed_real_control_closed":
        errors.append("Phase 5-H frozen status is invalid")
    parent_path = check_hash(status["phase5g_status"], errors, "Phase 5-G status")
    contract_path = check_hash(status["contract"], errors, "Phase 5-H contract")
    check_hash(status["rejected_clock_audit"], errors, "Phase 5-H clock audit")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    summary_path = check_hash(status["takeover"]["summary"], errors, "Phase 5-H summary")
    evidence_path = check_hash(status["takeover"]["evidence"], errors, "Phase 5-H evidence")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    acceptance = contract["acceptance"]
    if parent["status"] != "phase5g_simulated_takeover_passed_real_control_closed":
        errors.append("Phase 5-G prerequisite is not passed")
    if summary["status"] != "articulation_gate_passed" or not summary["gate_passed"]:
        errors.append("Phase 5-H formal gate is not passed")
    if summary["evidence_sha256"] != sha256(evidence_path):
        errors.append("Phase 5-H evidence hash differs from summary")
    if summary["direct_pose_updates_after_initialization"] != 0:
        errors.append("direct pose updates were used after scenario initialization")
    if summary["oracle_command_override_count"] != 0:
        errors.append("Oracle replaced a candidate command")
    if not summary["candidate_controls_articulation"] or summary["real_vehicle_control_allowed"]:
        errors.append("Phase 5-H authority boundary is invalid")
    if [item["name"] for item in summary["scenarios"]] != contract["scenarios"]:
        errors.append("Phase 5-H scenario membership drifted")

    totals = {"reached": 0, "supervisor_aborts": 0, "static_collision_count": 0, "dynamic_collision_count": 0, "solver_failures": 0}
    for scenario in summary["scenarios"]:
        name = scenario["name"]
        telemetry_path = summary_path.parent / scenario["telemetry"]
        if not telemetry_path.is_file() or sha256(telemetry_path) != scenario["telemetry_sha256"]:
            errors.append(f"telemetry missing or changed: {name}")
            continue
        with telemetry_path.open(newline="", encoding="ascii") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != scenario["steps"] or [int(row["step"]) for row in rows] != list(range(len(rows))):
            errors.append(f"non-exact telemetry sequence: {name}")
            continue
        pose = contract["scenario_poses"][name]
        initial = scenario["initial_state"]
        if math.hypot(initial["x_m"] - pose["start"][0], initial["y_m"] - pose["start"][1]) > 0.01:
            errors.append(f"settled articulation start drifted from contract: {name}")
        previous_x, previous_y, previous_v = initial["x_m"], initial["y_m"], initial["velocity_mps"]
        root_path_length = 0.0
        wheel_odometry = 0.0
        for row in rows:
            x, y, velocity = float(row["x_m"]), float(row["y_m"]), float(row["velocity_mps"])
            root_path_length += math.hypot(x - previous_x, y - previous_y)
            wheel_odometry += 0.5 * (previous_v + velocity) * DT
            previous_x, previous_y, previous_v = x, y, velocity
            expected_left = (float(row["target_velocity_mps"]) - float(row["omega_radps"]) * WHEEL_BASE_M * 0.5) / WHEEL_RADIUS_M
            expected_right = (float(row["target_velocity_mps"]) + float(row["omega_radps"]) * WHEEL_BASE_M * 0.5) / WHEEL_RADIUS_M
            if not close(row["left_target_radps"], expected_left, 1e-6) or not close(row["right_target_radps"], expected_right, 1e-6):
                errors.append(f"wheel target does not encode NMPC command: {name}")
                break
        terminal = rows[-1]
        goal = pose["goal"]
        recomputed = {
            "solver_failures": sum(int(row["solver_status"]) != 0 for row in rows),
            "static_collision_count": sum(int(row["static_collision"]) for row in rows),
            "dynamic_collision_count": sum(int(row["dynamic_collision"]) for row in rows),
            "wheel_command_applied_ratio": sum(int(row["wheel_command_applied"]) for row in rows) / len(rows),
            "physics_feedback_ratio": sum(int(row["physics_feedback"]) for row in rows) / len(rows),
            "candidate_valid_ratio_mean": sum(float(row["candidate_valid_ratio"]) for row in rows) / len(rows),
            "render_p95_ms": percentile(rows, "render_ms"),
            "candidate_pipeline_p95_ms": percentile(rows, "candidate_pipeline_ms"),
            "sensor_to_wheel_p95_ms": percentile(rows, "sensor_to_wheel_ms"),
            "path_error_p95_m": percentile(rows, "path_error_m"),
            "terminal_position_error_m": math.hypot(goal[0] - float(terminal["x_m"]), goal[1] - float(terminal["y_m"])),
            "terminal_yaw_error_rad": abs(wrap_angle(goal[2] - float(terminal["yaw_rad"]))),
            "root_travel_m": math.hypot(float(terminal["x_m"]) - initial["x_m"], float(terminal["y_m"]) - initial["y_m"]),
            "root_path_length_m": root_path_length,
            "wheel_odometry_distance_m": wheel_odometry,
            "wheel_odometry_ratio": root_path_length / max(wheel_odometry, 1e-9),
        }
        for metric, value in recomputed.items():
            if not close(scenario[metric], value):
                errors.append(f"summary metric drifted for {name}: {metric}")
        totals["reached"] += int(scenario["reached"])
        for field in ("supervisor_aborts", "static_collision_count", "dynamic_collision_count", "solver_failures"):
            totals[field] += int(scenario[field])
        if any(row["supervisor_decision"] != "allow" for row in rows):
            errors.append(f"supervisor abort found in passed scenario: {name}")
        if any(int(row["candidate_controls_articulation"]) != 1 for row in rows):
            errors.append(f"candidate did not own articulation command: {name}")
        if scenario["sensor_to_wheel_p95_ms"] > acceptance["sensor_to_wheel_p95_ms_max"]:
            errors.append(f"end-to-end latency gate failed: {name}")
        if not acceptance["wheel_odometry_ratio_min"] <= recomputed["wheel_odometry_ratio"] <= acceptance["wheel_odometry_ratio_max"]:
            errors.append(f"physics/control clock ratio failed: {name}")
        if scenario["path_error_p95_m"] > acceptance["path_error_p95_m_max"]:
            errors.append(f"path tracking gate failed: {name}")
        if scenario["terminal_position_error_m"] > acceptance["terminal_position_error_m_max"] or scenario["terminal_yaw_error_rad"] > acceptance["terminal_yaw_error_rad_max"]:
            errors.append(f"terminal pose gate failed: {name}")
        if scenario["root_travel_m"] < acceptance["minimum_root_travel_m"]:
            errors.append(f"articulation root did not travel: {name}")
        if name == "crossing_cart":
            encounter = sum(float(row["dynamic_center_distance_m"]) <= 2.2 for row in rows)
            if encounter != scenario["dynamic_encounter_frames"] or encounter < acceptance["dynamic_encounter_frames_min"]:
                errors.append("crossing_cart dynamic encounter drifted or is insufficient")

    expected = {
        "reached": acceptance["reached_scenarios"],
        "supervisor_aborts": acceptance["supervisor_aborts"],
        "static_collision_count": acceptance["static_collision_count"],
        "dynamic_collision_count": acceptance["dynamic_collision_count"],
        "solver_failures": acceptance["solver_failures"],
    }
    if totals != expected:
        errors.append(f"Phase 5-H aggregate gate failed: {totals}")
    authority = status["control_authority"]
    if not authority["candidate_controls_isaac_articulation"] or authority["real_vehicle_control_allowed"]:
        errors.append("Phase 5-H frozen authority exceeded simulation")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-H Isaac articulation takeover validation OK")
    print("4/4 PhysX scenarios reached; zero aborts, collisions and solver failures")
    print("Full RGB+depth-to-wheel p95 is below 50 ms; real control remains closed")


if __name__ == "__main__":
    main()
