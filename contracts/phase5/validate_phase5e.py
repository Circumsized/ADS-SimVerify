#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5e_status.json"


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


def validate_calibration_block(status, errors):
    summary_path = check_hash(status["real_calibration"]["summary"], errors, "calibration audit")
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "calibration_gate_blocked" or summary["gate_passed"]:
        errors.append("real calibration gate is not frozen as blocked")
    if summary["video_devices"]:
        errors.append("calibration audit unexpectedly contains an unreviewed video device")
    required_errors = {
        "no /dev/video* real camera or depth device is present",
        "no real camera calibration file was supplied",
    }
    if not required_errors.issubset(summary["errors"]):
        errors.append("real calibration blocker evidence is incomplete")
    if summary["control_promotion_allowed"]:
        errors.append("blocked calibration audit opened control promotion")


def validate_live(status, contract, errors):
    summary_path = check_hash(status["live_shadow"]["summary"], errors, "live summary")
    telemetry_path = check_hash(status["live_shadow"]["telemetry"], errors, "live telemetry")
    if not summary_path.is_file() or not telemetry_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with telemetry_path.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    acceptance = contract["live_isaac_shadow"]["acceptance"]
    frame_count = contract["live_isaac_shadow"]["frames"]
    if len(rows) != frame_count or summary["frames"] != frame_count:
        errors.append("Phase 5-E live evidence must contain exactly 1000 frames")
    if [int(row["source_frame_id"]) for row in rows] != list(range(frame_count)):
        errors.append("Phase 5-E live source frame ids are not exact")
    expected_modes = {
        "center_stop": 400,
        "side_go_left": 100,
        "side_go_right": 100,
        "far_go": 200,
        "absent_go": 200,
    }
    if Counter(row["dynamic_mode"] for row in rows) != expected_modes:
        errors.append("Phase 5-E live dynamic schedule drifted")
    metrics = {
        name: mean(rows, name)
        for name in (
            "occupied_iou",
            "free_iou",
            "false_free_rate",
            "false_occupied_rate",
        )
    }
    if not (
        metrics["occupied_iou"] >= acceptance["occupied_iou_mean_min"]
        and metrics["free_iou"] >= acceptance["free_iou_mean_min"]
        and metrics["false_free_rate"] <= acceptance["false_free_rate_mean_max"]
        and metrics["false_occupied_rate"] <= acceptance["false_occupied_rate_mean_max"]
    ):
        errors.append("Phase 5-E live perception metric gate failed")
    for name, value in metrics.items():
        if abs(summary["metrics"][name]["mean"] - value) > 1e-12:
            errors.append(f"live summary differs from telemetry: {name}")
    confusion = {
        "true_stop": sum(int(row["oracle_stop"]) and int(row["candidate_stop"]) for row in rows),
        "missed_stop": sum(int(row["oracle_stop"]) and not int(row["candidate_stop"]) for row in rows),
        "true_go": sum(not int(row["oracle_stop"]) and not int(row["candidate_stop"]) for row in rows),
        "false_stop": sum(not int(row["oracle_stop"]) and int(row["candidate_stop"]) for row in rows),
    }
    for name, value in confusion.items():
        if summary["stop_decision"][name] != value:
            errors.append(f"live STOP/GO summary differs: {name}")
    stop_recall = confusion["true_stop"] / max(confusion["true_stop"] + confusion["missed_stop"], 1)
    go_specificity = confusion["true_go"] / max(confusion["true_go"] + confusion["false_stop"], 1)
    if stop_recall < acceptance["stop_recall_min"] or go_specificity < acceptance["go_specificity_min"]:
        errors.append("Phase 5-E live dynamic gate failed")
    valid_mean = mean(rows, "candidate_valid_ratio")
    latency_p95 = percentile(rows, "latency_ms", 0.95)
    if valid_mean < acceptance["candidate_valid_ratio_mean_min"]:
        errors.append("Phase 5-E live validity gate failed")
    if latency_p95 > acceptance["latency_p95_ms_max"]:
        errors.append("Phase 5-E live latency gate failed")
    if abs(summary["candidate_valid_ratio"]["mean"] - valid_mean) > 1e-12:
        errors.append("live valid ratio summary drifted")
    if abs(summary["latency_ms"]["p95"] - latency_p95) > 1e-12:
        errors.append("live latency summary drifted")
    if summary["status"] != "live_shadow_passed" or not summary["gate_passed"]:
        errors.append("Phase 5-E live gate is not frozen as passed")
    if summary["telemetry_sha256"] != sha256(telemetry_path):
        errors.append("live telemetry differs from its summary")
    if summary["control_output_declared"] or summary["candidate_controls_vehicle"] or summary["control_promotion_allowed"]:
        errors.append("Phase 5-E live shadow was granted control authority")


def validate_topology(status, errors):
    path = check_hash(status["implementation"]["live_dataflow"], errors, "live dataflow")
    if not path.is_file():
        return
    topology = yaml.safe_load(path.read_text(encoding="utf-8"))
    serialized = json.dumps(topology)
    outputs = [output for node in topology["nodes"] for output in node.get("outputs", [])]
    if "control_cmd" in outputs or "control_cmd" in serialized or "fast_brain_nmpc" in serialized:
        errors.append("Phase 5-E live topology has a control output, edge or consumer")


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5e_live_passed_real_calibration_blocked_control_closed":
        errors.append("Phase 5-E frozen status is invalid")
    check_hash(status["phase5d_status"], errors, "Phase 5-D status")
    contract_path = check_hash(status["contract"], errors, "Phase 5-E contract")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    validate_calibration_block(status, errors)
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if not contract["promotion"]["requires_both_gates"]:
            errors.append("Phase 5-E no longer requires both gates")
        validate_live(status, contract, errors)
    validate_topology(status, errors)
    gate = status["gate"]
    if gate["real_calibration_passed"] or not gate["live_isaac_shadow_passed"]:
        errors.append("Phase 5-E partial gate state drifted")
    if gate["phase5e_passed"] or gate["control_promotion_allowed"]:
        errors.append("blocked Phase 5-E incorrectly opened promotion")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-E live Isaac metric-depth shadow validation OK")
    print("1000 exact live frames passed; real-camera calibration remains blocked")
    print("Phase 5-E and control promotion remain closed")


if __name__ == "__main__":
    main()
