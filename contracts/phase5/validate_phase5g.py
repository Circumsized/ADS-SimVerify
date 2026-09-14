#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5g_status.json"


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
    return abs(float(actual) - float(expected)) <= tolerance


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5g_simulated_takeover_passed_real_control_closed":
        errors.append("Phase 5-G frozen status is invalid")
    parent_path = check_hash(status["phase5f_status"], errors, "Phase 5-F status")
    contract_path = check_hash(status["contract"], errors, "Phase 5-G contract")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    summary_path = check_hash(status["takeover"]["summary"], errors, "Phase 5-G summary")
    evidence_path = check_hash(status["takeover"]["evidence"], errors, "Phase 5-G evidence")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    acceptance = contract["acceptance"]
    if parent["status"] != "phase5f_perception_nmpc_shadow_passed_control_closed":
        errors.append("Phase 5-F parent gate is not passed and control-closed")
    if summary["status"] != "takeover_gate_passed" or not summary["gate_passed"]:
        errors.append("Phase 5-G summary gate is not passed")
    if summary["evidence_sha256"] != sha256(evidence_path):
        errors.append("Phase 5-G evidence hash differs from summary")
    if summary["oracle_command_override_count"] != 0:
        errors.append("Oracle supervisor replaced a candidate command")
    if not summary["candidate_controls_simulation"] or summary["real_vehicle_control_allowed"]:
        errors.append("Phase 5-G authority boundary is invalid")
    if [item["name"] for item in summary["scenarios"]] != contract["scenarios"]:
        errors.append("Phase 5-G scenario order or membership drifted")

    totals = {
        "reached": 0,
        "supervisor_aborts": 0,
        "static_collision_count": 0,
        "dynamic_collision_count": 0,
        "solver_failures": 0,
    }
    for scenario in summary["scenarios"]:
        telemetry_path = summary_path.parent / scenario["telemetry"]
        if not telemetry_path.is_file():
            errors.append(f"missing telemetry: {telemetry_path}")
            continue
        if sha256(telemetry_path) != scenario["telemetry_sha256"]:
            errors.append(f"telemetry hash mismatch: {scenario['name']}")
        with telemetry_path.open(newline="", encoding="ascii") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != scenario["steps"] or [int(row["step"]) for row in rows] != list(range(len(rows))):
            errors.append(f"non-exact telemetry sequence: {scenario['name']}")
            continue
        totals["reached"] += int(scenario["reached"])
        for field in ("supervisor_aborts", "static_collision_count", "dynamic_collision_count", "solver_failures"):
            totals[field] += int(scenario[field])
        recomputed = {
            "solver_failures": sum(int(row["solver_status"]) != 0 for row in rows),
            "static_collision_count": sum(int(row["static_collision"]) for row in rows),
            "dynamic_collision_count": sum(int(row["dynamic_collision"]) for row in rows),
            "command_applied_ratio": mean(rows, "command_applied"),
            "candidate_valid_ratio_mean": mean(rows, "candidate_valid_ratio"),
            "candidate_pipeline_p95_ms": percentile(rows, "candidate_pipeline_ms", 0.95),
            "sensor_to_command_p95_ms": percentile(rows, "sensor_to_command_ms", 0.95),
            "render_p95_ms": percentile(rows, "render_ms", 0.95),
            "path_error_p95_m": percentile(rows, "path_error_m", 0.95),
        }
        for name, value in recomputed.items():
            if not close(scenario[name], value):
                errors.append(f"summary metric drifted for {scenario['name']}: {name}")
        if any(row["supervisor_decision"] != "allow" for row in rows):
            errors.append(f"supervisor abort found in passed scenario: {scenario['name']}")
        if any(int(row["candidate_controls_simulation"]) != 1 for row in rows):
            errors.append(f"candidate did not own simulated commands: {scenario['name']}")
        if scenario["command_applied_ratio"] < acceptance["candidate_command_applied_ratio_min"]:
            errors.append(f"candidate command application gate failed: {scenario['name']}")
        if scenario["candidate_valid_ratio_mean"] < acceptance["candidate_valid_ratio_mean_min"]:
            errors.append(f"candidate validity gate failed: {scenario['name']}")
        if scenario["candidate_pipeline_p95_ms"] > acceptance["candidate_pipeline_p95_ms_max"]:
            errors.append(f"candidate latency gate failed: {scenario['name']}")
        if scenario["path_error_p95_m"] > acceptance["path_error_p95_m_max"]:
            errors.append(f"path tracking gate failed: {scenario['name']}")
        if scenario["terminal_position_error_m"] > acceptance["terminal_position_error_m_max"]:
            errors.append(f"terminal position gate failed: {scenario['name']}")
        if scenario["terminal_yaw_error_rad"] > acceptance["terminal_yaw_error_rad_max"]:
            errors.append(f"terminal yaw gate failed: {scenario['name']}")
        if scenario["name"] == "crossing_cart":
            encounter = sum(float(row["dynamic_center_distance_m"]) <= 2.2 for row in rows)
            if encounter != scenario["dynamic_encounter_frames"]:
                errors.append("crossing_cart encounter count drifted")
            if encounter < acceptance["dynamic_encounter_frames_min"]:
                errors.append("crossing_cart did not exercise the dynamic encounter")

    expected_totals = {
        "reached": acceptance["reached_scenarios"],
        "supervisor_aborts": acceptance["supervisor_aborts"],
        "static_collision_count": acceptance["static_collision_count"],
        "dynamic_collision_count": acceptance["dynamic_collision_count"],
        "solver_failures": acceptance["solver_failures"],
    }
    if totals != expected_totals:
        errors.append(f"Phase 5-G aggregate gate failed: {totals}")
    authority = status["control_authority"]
    if not authority["candidate_controls_simulation"]:
        errors.append("candidate simulated control authority was not frozen")
    if authority["oracle_command_override_count"] or authority["real_vehicle_control_allowed"]:
        errors.append("Phase 5-G authority exceeded simulation-only allow-or-abort")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-G controlled simulation takeover validation OK")
    print("4/4 scenarios reached; zero Oracle aborts, collisions and solver failures")
    print("Candidate controls kinematic simulation only; real control remains closed")


if __name__ == "__main__":
    main()
