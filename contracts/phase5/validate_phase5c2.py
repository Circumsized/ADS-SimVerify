#!/usr/bin/env python3
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5c2_status.json"
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"


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


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    thresholds = phase3["phase4_perception_gate"]
    errors = []
    if status["status"] != "phase5c2_geometry_frozen_training_open_control_closed":
        errors.append("Phase 5-C2 frozen status is invalid")
    phase5c_path = check_hash(status["phase5c_status"], errors, "Phase 5-C status")
    if phase5c_path.is_file():
        phase5c = json.loads(phase5c_path.read_text(encoding="utf-8"))
        if not phase5c["measured_result"]["model_training_allowed"] is False:
            errors.append("Phase 5-C2 does not descend from the frozen geometry blocker")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    oracle_manifest_path = check_hash(
        status["perception_oracle"]["manifest"], errors, "perception Oracle manifest"
    )
    oracle_archive_path = check_hash(
        status["perception_oracle"]["archive"], errors, "perception Oracle archive"
    )
    check_hash(status["perception_oracle"]["preview"], errors, "perception Oracle preview")
    summary_path = check_hash(status["evidence"]["summary"], errors, "Phase 5-C2 summary")
    telemetry_path = check_hash(status["evidence"]["telemetry"], errors, "Phase 5-C2 telemetry")

    if oracle_manifest_path.is_file() and oracle_archive_path.is_file():
        oracle = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
        if oracle["status"] != "perception_scoring_only" or oracle["control_authority"]:
            errors.append("perception Oracle was granted control authority")
        if oracle["height_band_m"] != [0.02, 0.35]:
            errors.append("perception Oracle height band drifted")
        if oracle["archive_sha256"] != sha256(oracle_archive_path):
            errors.append("perception Oracle archive differs from its manifest")

    if summary_path.is_file() and telemetry_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with telemetry_path.open(newline="", encoding="ascii") as source:
            rows = list(csv.DictReader(source))
        if summary["status"] != "geometry_upper_bound_passed_model_training_allowed":
            errors.append("Phase 5-C2 upper bound is not frozen as passed")
        if len(rows) != 1000 or summary["frame_count"] != 1000:
            errors.append("Phase 5-C2 evidence must contain exactly 1000 frames")
        if [int(row["source_frame_id"]) for row in rows] != list(range(1000)):
            errors.append("Phase 5-C2 frame ids are not exact and contiguous")
        expected_modes = {
            "center_stop": 400,
            "side_go_left": 100,
            "side_go_right": 100,
            "far_go": 200,
            "absent_go": 200,
        }
        if Counter(row["dynamic_mode"] for row in rows) != expected_modes:
            errors.append("Phase 5-C2 dynamic schedule drifted")
        measured_stop = {
            "true_stop": sum(int(row["oracle_stop"]) and int(row["depth_gt_stop"]) for row in rows),
            "missed_stop": sum(int(row["oracle_stop"]) and not int(row["depth_gt_stop"]) for row in rows),
            "true_go": sum(not int(row["oracle_stop"]) and not int(row["depth_gt_stop"]) for row in rows),
            "false_stop": sum(not int(row["oracle_stop"]) and int(row["depth_gt_stop"]) for row in rows),
        }
        if measured_stop != {"true_stop": 400, "missed_stop": 0, "true_go": 600, "false_stop": 0}:
            errors.append("Phase 5-C2 dynamic STOP/GO gate failed")
        for name, value in measured_stop.items():
            if summary["stop_decision"][name] != value:
                errors.append(f"Phase 5-C2 stop metric differs from telemetry: {name}")
        measured_metrics = {
            "occupied_iou": mean(rows, "depth_gt_occupied_iou"),
            "free_iou": mean(rows, "depth_gt_free_iou"),
            "false_free_rate": mean(rows, "depth_gt_false_free_rate"),
            "false_occupied_rate": mean(rows, "depth_gt_false_occupied_rate"),
        }
        for name, value in measured_metrics.items():
            if abs(summary["gt_depth_lift"][name]["mean"] - value) > 1e-12:
                errors.append(f"Phase 5-C2 summary metric differs from telemetry: {name}")
        if measured_metrics["occupied_iou"] < thresholds["bc_occupied_iou_mean_min"]:
            errors.append("occupied IoU remains below the frozen gate")
        if measured_metrics["free_iou"] < thresholds["bc_free_iou_mean_min"]:
            errors.append("free IoU remains below the frozen gate")
        if measured_metrics["false_free_rate"] > thresholds["bc_false_free_rate_mean_max"]:
            errors.append("false-free remains above the frozen gate")
        if measured_metrics["false_occupied_rate"] > thresholds["bc_false_occupied_rate_mean_max"]:
            errors.append("false-occupied remains above the frozen gate")
        if summary["depth_latency_ms"]["p95"] > thresholds["latency_p95_ms_max"]:
            errors.append("Phase 5-C2 depth geometry exceeds the latency gate")
        gate = summary["gate"]
        if not gate["upper_bound_passed"] or not gate["model_training_allowed"]:
            errors.append("passed teacher upper bound did not open model training")
        if gate["control_promotion_allowed"]:
            errors.append("Phase 5-C2 incorrectly promoted perception to control")
        authority = summary["control_authority"]
        if any(
            authority[name]
            for name in (
                "control_output_declared",
                "dynamic_obstacle_affects_control",
                "perception_oracle_controls_vehicle",
            )
        ):
            errors.append("Phase 5-C2 shadow evidence entered the control loop")
        if summary["telemetry_sha256"] != sha256(telemetry_path):
            errors.append("Phase 5-C2 telemetry differs from its summary")
        for item in summary["evidence"]:
            path = summary_path.parent / item["path"]
            if not path.is_file() or sha256(path) != item["sha256"]:
                errors.append(f"Phase 5-C2 evidence drifted: {item['path']}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-C2 depth-to-BEV geometry validation OK")
    print("1000 exact frames; teacher perception and dynamic STOP/GO gates passed")
    print("Warehouse model training is allowed; control promotion remains closed")


if __name__ == "__main__":
    main()
