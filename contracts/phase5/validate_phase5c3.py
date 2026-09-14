#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5c3_status.json"
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


def percentile(rows, field, quantile):
    values = sorted(float(row[field]) for row in rows)
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def validate_dataset(status, contract, errors):
    summary_path = check_hash(status["dataset"]["summary"], errors, "dataset summary")
    manifest_path = check_hash(status["dataset"]["manifest"], errors, "dataset manifest")
    if not summary_path.is_file() or not manifest_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with manifest_path.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    expected = {"train": 1200, "validation": 300}
    counts = Counter(row["split"] for row in rows)
    if counts != expected or len(rows) != sum(expected.values()):
        errors.append("Phase 5-C3 dataset split sizes drifted")
    for split, count in expected.items():
        ids = [int(row["frame_id"]) for row in rows if row["split"] == split]
        if ids != list(range(count)):
            errors.append(f"Phase 5-C3 {split} frame ids are not exact")
    if summary["status"] != "complete" or any(summary["missing_classes"].values()):
        errors.append("Phase 5-C3 dataset does not cover all warehouse_nav14 classes")
    if summary["phase5_gate_frames_used"] or summary["phase4_smoke_frames_used"]:
        errors.append("frozen gate/smoke frames leaked into Phase 5-C3 training")
    if summary["frame_manifest_sha256"] != sha256(manifest_path):
        errors.append("dataset manifest differs from its summary")
    for split in expected:
        pixels = summary["splits"][split]["class_pixels"]
        if len(pixels) != 14 or any(int(value) <= 0 for value in pixels.values()):
            errors.append(f"Phase 5-C3 {split} class coverage is incomplete")

    oracle_path = ROOT / summary["perception_oracle_manifest"]
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    min_x, _, min_y, _ = oracle["bounds_xy_m"]
    blocks = {"train": set(), "validation": set()}
    for row in rows:
        block = (
            math.floor((float(row["x_m"]) - min_x) / 2.0),
            math.floor((float(row["y_m"]) - min_y) / 2.0),
        )
        blocks[row["split"]].add(block)
    if blocks["train"] & blocks["validation"]:
        errors.append("train/validation world-space blocks overlap")

    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    closed_loop_summary = ROOT / phase5a["closed_loop"]["summary"]["path"]
    gate_points = []
    for csv_path in closed_loop_summary.parent.glob("*.csv"):
        with csv_path.open(newline="", encoding="ascii") as source:
            gate_points.extend(
                (float(row["x_m"]), float(row["y_m"])) for row in csv.DictReader(source)
            )
    exclusion = contract["dataset"]["phase5_gate_trajectory_exclusion_m"]
    for row in rows:
        x, y = float(row["x_m"]), float(row["y_m"])
        if min(math.hypot(x - gx, y - gy) for gx, gy in gate_points) < exclusion - 1e-9:
            errors.append("dataset pose violates the frozen gate trajectory exclusion")
            break


def validate_training(status, contract, thresholds, errors):
    summary_path = check_hash(status["training"]["summary"], errors, "training summary")
    check_hash(status["training"]["checkpoint"], errors, "training checkpoint")
    artifact_model = check_hash(status["training"]["onnx"], errors, "training ONNX")
    deployed_model = check_hash(status["candidate"]["model"], errors, "deployed candidate")
    manifest_path = check_hash(status["candidate"]["manifest"], errors, "candidate manifest")
    if not summary_path.is_file() or not manifest_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if summary["status"] != "candidate_trained_shadow_only":
        errors.append("training output is not frozen as a shadow-only candidate")
    expected = contract["training"]
    if (
        summary["train_frames"] != contract["dataset"]["train_frames"]
        or summary["validation_frames"] != contract["dataset"]["validation_frames"]
        or summary["epochs"] != expected["epochs"]
        or summary["batch_size"] != expected["batch_size"]
        or len(summary["history"]) != expected["epochs"]
    ):
        errors.append("training run does not match the frozen contract")
    if summary["device"] != "cuda" or "RTX 3060" not in summary["gpu"]:
        errors.append("Phase 5-C3 was not trained on the frozen CUDA hardware boundary")
    validation = summary["onnx_validation"]
    if validation["input_shape"] != [1, 3, 240, 320] or validation["output_shape"] != [1, 14, 240, 320]:
        errors.append("candidate ONNX tensor contract drifted")
    if validation["latency_ms"]["p95"] > thresholds["latency_p95_ms_max"]:
        errors.append("candidate ONNX validation latency exceeds the frozen gate")
    if artifact_model.is_file() and deployed_model.is_file() and sha256(artifact_model) != sha256(deployed_model):
        errors.append("deployed candidate differs from the selected training artifact")
    if manifest["model_sha256"] != sha256(deployed_model):
        errors.append("candidate manifest model hash drifted")
    if manifest["training_summary_sha256"] != sha256(summary_path):
        errors.append("candidate manifest training hash drifted")
    if summary["control_promotion_allowed"] or manifest["control_promotion_allowed"]:
        errors.append("training incorrectly promoted the candidate to control")


def validate_shadow(status, thresholds, errors):
    summary_path = check_hash(status["shadow"]["summary"], errors, "shadow summary")
    telemetry_path = check_hash(status["shadow"]["telemetry"], errors, "shadow telemetry")
    if not summary_path.is_file() or not telemetry_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with telemetry_path.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1000 or summary["frame_count"] != 1000:
        errors.append("Phase 5-C3 shadow evidence must contain exactly 1000 frames")
    if [int(row["source_frame_id"]) for row in rows] != list(range(1000)):
        errors.append("Phase 5-C3 shadow frame ids are not exact and contiguous")
    expected_modes = {
        "center_stop": 400,
        "side_go_left": 100,
        "side_go_right": 100,
        "far_go": 200,
        "absent_go": 200,
    }
    if Counter(row["dynamic_mode"] for row in rows) != expected_modes:
        errors.append("Phase 5-C3 dynamic schedule drifted")
    metrics = {
        name: mean(rows, f"candidate_{name}")
        for name in (
            "occupied_iou",
            "free_iou",
            "false_free_rate",
            "false_occupied_rate",
        )
    }
    gates = (
        metrics["occupied_iou"] >= thresholds["bc_occupied_iou_mean_min"],
        metrics["free_iou"] >= thresholds["bc_free_iou_mean_min"],
        metrics["false_free_rate"] <= thresholds["bc_false_free_rate_mean_max"],
        metrics["false_occupied_rate"] <= thresholds["bc_false_occupied_rate_mean_max"],
    )
    if not all(gates):
        errors.append("candidate metric gate failed on independently recomputed telemetry")
    for name, value in metrics.items():
        if abs(summary["methods"]["candidate"][name]["mean"] - value) > 1e-12:
            errors.append(f"candidate summary metric differs from telemetry: {name}")
    confusion = {
        "true_stop": sum(int(row["oracle_stop"]) and int(row["candidate_stop"]) for row in rows),
        "missed_stop": sum(int(row["oracle_stop"]) and not int(row["candidate_stop"]) for row in rows),
        "true_go": sum(not int(row["oracle_stop"]) and not int(row["candidate_stop"]) for row in rows),
        "false_stop": sum(not int(row["oracle_stop"]) and int(row["candidate_stop"]) for row in rows),
    }
    if confusion != {"true_stop": 400, "missed_stop": 0, "true_go": 599, "false_stop": 1}:
        errors.append("candidate dynamic STOP/GO evidence drifted")
    for name, value in confusion.items():
        if summary["stop_decision"]["candidate"][name] != value:
            errors.append(f"candidate stop summary differs from telemetry: {name}")
    valid_mean = mean(rows, "candidate_valid_ratio")
    latency_p95 = percentile(rows, "candidate_total_latency_ms", 0.95)
    if valid_mean < thresholds["perception_valid_ratio_min"]:
        errors.append("candidate valid ratio is below the frozen gate")
    if latency_p95 > thresholds["latency_p95_ms_max"]:
        errors.append("candidate total latency exceeds the frozen gate")
    if abs(summary["latency_ms"]["candidate_total"]["p95"] - latency_p95) > 1e-12:
        errors.append("candidate latency summary differs from telemetry")
    if summary["status"] != "candidate_shadow_passed" or not summary["gate"]["shadow_gate_passed"]:
        errors.append("candidate shadow gate is not frozen as passed")
    if summary["gate"]["control_promotion_allowed"]:
        errors.append("shadow gate incorrectly promoted the candidate to control")
    authority = summary["control_authority"]
    if authority["candidate_controls_vehicle"] or authority["control_output_declared"]:
        errors.append("candidate shadow output entered the control loop")
    if summary["telemetry_sha256"] != sha256(telemetry_path):
        errors.append("shadow telemetry differs from its summary")
    for item in summary["evidence"]:
        path = summary_path.parent / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            errors.append(f"Phase 5-C3 evidence drifted: {item['path']}")


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    thresholds = phase3["phase4_perception_gate"]
    errors = []
    if status["status"] != "phase5c3_candidate_shadow_passed_control_closed":
        errors.append("Phase 5-C3 frozen status is invalid")
    check_hash(status["phase5c2_status"], errors, "Phase 5-C2 status")
    contract_path = check_hash(status["contract"], errors, "Phase 5-C3 contract")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract["promotion"]["control_promotion_allowed"]:
            errors.append("Phase 5-C3 contract opened control promotion")
        validate_dataset(status, contract, errors)
        validate_training(status, contract, thresholds, errors)
    validate_shadow(status, thresholds, errors)
    if status["control_authority"]["control_promotion_allowed"]:
        errors.append("Phase 5-C3 status opened control promotion")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-C3 warehouse_nav14 adaptation validation OK")
    print("1500 disjoint dataset frames; CUDA training and 1000-frame shadow gate passed")
    print("Candidate remains shadow-only; Oracle NMPC retains sole control authority")


if __name__ == "__main__":
    main()
