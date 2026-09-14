#!/usr/bin/env python3
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5c_status.json"


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


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5c_gt_upper_bound_frozen_geometry_blocked":
        errors.append("Phase 5-C status does not preserve the geometry blocker")
    phase5b_path = check_hash(status["phase5b_status"], errors, "Phase 5-B status")
    if phase5b_path.is_file():
        phase5b = json.loads(phase5b_path.read_text(encoding="utf-8"))
        if phase5b["status"] != "phase5b_shadow_frozen_candidate_rejected":
            errors.append("Phase 5-C does not descend from rejected Phase 5-B candidate")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    summary_path = check_hash(status["upper_bound_evidence"]["summary"], errors, "upper-bound summary")
    telemetry_path = check_hash(status["upper_bound_evidence"]["telemetry"], errors, "upper-bound telemetry")

    if summary_path.is_file() and telemetry_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with telemetry_path.open(newline="", encoding="ascii") as source:
            rows = list(csv.DictReader(source))
        if summary["status"] != "gt_upper_bound_failed_model_training_blocked":
            errors.append("GT upper bound was not frozen as failed")
        if len(rows) != summary["frame_count"] or len(rows) != 1000:
            errors.append("dynamic upper-bound evidence must contain exactly 1000 frames")
        if [int(row["source_frame_id"]) for row in rows] != list(range(1000)):
            errors.append("dynamic frame ids are not exact and contiguous")
        expected_modes = {
            "center_stop": 400,
            "side_go_left": 100,
            "side_go_right": 100,
            "far_go": 200,
            "absent_go": 200,
        }
        if Counter(row["dynamic_mode"] for row in rows) != expected_modes:
            errors.append("dynamic STOP/GO schedule differs from the frozen gate")
        stopping = summary["stop_decision"]
        measured = {
            "true_stop": sum(int(row["oracle_stop"]) and int(row["depth_gt_stop"]) for row in rows),
            "missed_stop": sum(int(row["oracle_stop"]) and not int(row["depth_gt_stop"]) for row in rows),
            "true_go": sum(not int(row["oracle_stop"]) and not int(row["depth_gt_stop"]) for row in rows),
            "false_stop": sum(not int(row["oracle_stop"]) and int(row["depth_gt_stop"]) for row in rows),
        }
        for name, value in measured.items():
            if stopping[name] != value:
                errors.append(f"stop decision {name} differs from telemetry")
        if measured != {"true_stop": 400, "missed_stop": 0, "true_go": 600, "false_stop": 0}:
            errors.append("dynamic decision upper bound did not pass the balanced gate")
        gate = summary["gate"]
        if not gate["stop_decision_passed"]:
            errors.append("dynamic stop decision gate should pass")
        if gate["perception_metrics_passed"] or gate["upper_bound_passed"]:
            errors.append("failed pixel/BEV upper bound was incorrectly opened")
        if gate["model_training_allowed"]:
            errors.append("warehouse model training was allowed below the GT upper bound")
        authority = summary["control_authority"]
        if authority["control_output_declared"] or authority["dynamic_obstacle_affects_control"]:
            errors.append("counterfactual dynamic obstacle entered the control loop")
        if summary["telemetry_sha256"] != sha256(telemetry_path):
            errors.append("upper-bound telemetry hash differs from summary")
        for item in summary["evidence"]:
            path = summary_path.parent / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                errors.append(f"upper-bound evidence drifted: {item['path']}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-C dynamic GT upper-bound validation OK")
    print("STOP/GO decision: 400/400 positive and 600/600 negative frames correct")
    print("Pixel/BEV upper bound failed; model training and control promotion remain blocked")


if __name__ == "__main__":
    main()
