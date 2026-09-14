#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5a_status.json"


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


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def close(actual, expected, tolerance=1e-6):
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5a_oracle_nmpc_frozen":
        errors.append("Phase 5-A status is not frozen")

    phase4_path = check_hash(status["phase4_status"], errors, "Phase 4 status")
    if phase4_path.is_file():
        phase4 = json.loads(phase4_path.read_text(encoding="utf-8"))
        if phase4["status"] != "p4_ab_frozen_p4c_closed":
            errors.append("Phase 5-A does not descend from the frozen Phase 4 baseline")

    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    solver_json_path = check_hash(status["generated_solver"]["json"], errors, "generated solver JSON")
    check_hash(status["generated_solver"]["shared_library"], errors, "generated solver library")
    if solver_json_path.is_file():
        solver = json.loads(solver_json_path.read_text(encoding="utf-8"))
        constraints = solver["constraints"]
        dims = solver["dims"]
        if dims["nbx"] != 1 or dims["nbx_e"] != 1:
            errors.append("generated solver does not bound velocity across the horizon")
        expected = {
            "idxbx": [3],
            "lbx": [0.0],
            "ubx": [0.8],
            "idxbx_e": [3],
            "lbx_e": [0.0],
            "ubx_e": [0.8],
        }
        for field, value in expected.items():
            if constraints[field] != value:
                errors.append(f"generated solver velocity contract differs: {field}")

    oracle_manifest_path = check_hash(status["oracle_map"]["manifest"], errors, "oracle manifest")
    oracle_archive_path = check_hash(status["oracle_map"]["archive"], errors, "oracle archive")
    check_hash(status["oracle_map"]["preview"], errors, "oracle preview")
    if oracle_manifest_path.is_file():
        oracle = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
        if oracle["archive_sha256"] != sha256(oracle_archive_path):
            errors.append("oracle manifest archive hash mismatch")
        if oracle["source_overlay_sha256"] != phase4["p4_a_semantic_overlay"]["overlay_sha256"]:
            errors.append("oracle map was not built from the frozen Phase 4 overlay")
        footprint = oracle["robot_footprint"]
        if footprint["collision_check"] != "yaw-aware rectangle sampled against raw occupancy":
            errors.append("oracle collision acceptance is not yaw-aware")

    summary_path = check_hash(status["closed_loop"]["summary"], errors, "closed-loop summary")
    evidence_path = check_hash(status["closed_loop"]["evidence"], errors, "closed-loop evidence")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        acceptance = summary["acceptance"]
        if summary["status"] != "passed":
            errors.append("closed-loop summary did not pass")
        if summary["learned_perception_in_control_loop"] is not False:
            errors.append("learned perception entered the Oracle control loop")
        if summary["oracle_manifest_sha256"] != sha256(oracle_manifest_path):
            errors.append("closed loop references a different oracle manifest")
        if summary["oracle_archive_sha256"] != sha256(oracle_archive_path):
            errors.append("closed loop references a different oracle archive")
        if summary["evidence_sha256"] != sha256(evidence_path):
            errors.append("closed-loop evidence hash differs from summary")
        scenarios = {item["name"]: item for item in summary["scenarios"]}
        if set(scenarios) != {"straight_aisle", "diagonal_turn", "pallet_detour"}:
            errors.append("closed-loop scenario set differs from the frozen gate")
        for name, scenario in scenarios.items():
            telemetry_path = summary_path.parent / scenario["telemetry"]
            if not telemetry_path.is_file() or sha256(telemetry_path) != scenario["telemetry_sha256"]:
                errors.append(f"{name} telemetry is missing or has drifted")
                continue
            with telemetry_path.open(newline="", encoding="ascii") as source:
                rows = list(csv.DictReader(source))
            if len(rows) != scenario["steps"] or not rows:
                errors.append(f"{name} telemetry row count differs from summary")
                continue
            collisions = sum(int(row["collision"]) for row in rows)
            failures = sum(int(row["solver_status"]) != 0 for row in rows)
            omega = [abs(float(row["omega_radps"])) for row in rows]
            acceleration = [abs(float(row["acceleration_mps2"])) for row in rows]
            latency = [float(row["solve_ms"]) for row in rows]
            path_error = [float(row["path_error_m"]) for row in rows]
            obstacle_h = [float(row["minimum_predicted_obstacle_h"]) for row in rows]
            if collisions != 0 or failures != 0:
                errors.append(f"{name} contains a collision or solver failure")
            if max(omega) > acceptance["max_abs_omega_radps"]:
                errors.append(f"{name} exceeds angular velocity bounds")
            if max(acceleration) > acceptance["max_abs_acceleration_mps2"]:
                errors.append(f"{name} exceeds acceleration bounds")
            if percentile(latency, 0.95) > acceptance["solve_p95_ms_max"]:
                errors.append(f"{name} exceeds the solve latency budget")
            if percentile(path_error, 0.95) > acceptance["path_error_p95_m_max"]:
                errors.append(f"{name} exceeds the path tracking error gate")
            if min(obstacle_h) < acceptance["minimum_predicted_obstacle_h"]:
                errors.append(f"{name} violates the NMPC obstacle constraint")
            if not close(percentile(latency, 0.95), scenario["solve_latency_ms"]["p95"]):
                errors.append(f"{name} latency summary differs from telemetry")
            if not close(percentile(path_error, 0.95), scenario["path_error_m"]["p95"]):
                errors.append(f"{name} tracking summary differs from telemetry")
            if not scenario["passed"] or not scenario["reached"]:
                errors.append(f"{name} was not accepted")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-A Oracle NMPC frozen contract validation OK")
    print("3/3 scenarios reached, zero collisions, zero solver failures")
    print("Velocity prediction is constrained to [0.0, 0.8] m/s across the horizon")


if __name__ == "__main__":
    main()
