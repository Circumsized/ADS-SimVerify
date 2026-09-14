#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5d_status.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory):
    digest = hashlib.sha256()
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(directory)).encode("utf-8") + b"\0")
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


def validate_robustness(status, contract, errors):
    summary_path = check_hash(status["robustness"]["summary"], errors, "robustness summary")
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["status"] != "robustness_passed" or not summary["all_profiles_passed"]:
        errors.append("Phase 5-D multi-seed robustness is not frozen as passed")
    if summary["contract_sha256"] != sha256(ROOT / summary["contract"]):
        errors.append("robustness evidence contract hash drifted")
    if summary["candidate_sha256"] != sha256(ROOT / summary["candidate"]):
        errors.append("robustness candidate hash drifted")
    expected_profiles = {
        item["name"]: item["seed"] for item in contract["robustness"]["profiles"]
    }
    if {item["name"]: item["seed"] for item in summary["profiles"]} != expected_profiles:
        errors.append("Phase 5-D profile names or seeds drifted")
    acceptance = contract["robustness"]["acceptance"]
    expected_modes = {
        "center_stop": 400,
        "side_go_left": 100,
        "side_go_right": 100,
        "far_go": 200,
        "absent_go": 200,
    }
    for item in summary["profiles"]:
        reference = status["robustness"]["telemetry"][item["name"]]
        telemetry_path = check_hash(reference, errors, f"{item['name']} telemetry")
        if not telemetry_path.is_file():
            continue
        with telemetry_path.open(newline="", encoding="ascii") as source:
            rows = list(csv.DictReader(source))
        if len(rows) != 1000 or [int(row["source_frame_id"]) for row in rows] != list(range(1000)):
            errors.append(f"{item['name']} does not contain 1000 exact frames")
        if Counter(row["dynamic_mode"] for row in rows) != expected_modes:
            errors.append(f"{item['name']} dynamic schedule drifted")
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
            errors.append(f"{item['name']} independently recomputed metric gate failed")
        for name, value in metrics.items():
            if abs(item["metrics"][name]["mean"] - value) > 1e-12:
                errors.append(f"{item['name']} summary differs from telemetry: {name}")
        confusion = {
            "true_stop": sum(int(row["oracle_stop"]) and int(row["candidate_stop"]) for row in rows),
            "missed_stop": sum(int(row["oracle_stop"]) and not int(row["candidate_stop"]) for row in rows),
            "true_go": sum(not int(row["oracle_stop"]) and not int(row["candidate_stop"]) for row in rows),
            "false_stop": sum(not int(row["oracle_stop"]) and int(row["candidate_stop"]) for row in rows),
        }
        for name, value in confusion.items():
            if item["stop_decision"][name] != value:
                errors.append(f"{item['name']} STOP/GO summary differs: {name}")
        stop_recall = confusion["true_stop"] / max(confusion["true_stop"] + confusion["missed_stop"], 1)
        go_specificity = confusion["true_go"] / max(confusion["true_go"] + confusion["false_stop"], 1)
        if stop_recall < acceptance["stop_recall_min"] or go_specificity < acceptance["go_specificity_min"]:
            errors.append(f"{item['name']} dynamic gate failed")
        valid_mean = mean(rows, "candidate_valid_ratio")
        latency_p95 = percentile(rows, "candidate_latency_ms", 0.95)
        if valid_mean < acceptance["candidate_valid_ratio_mean_min"]:
            errors.append(f"{item['name']} valid ratio gate failed")
        if latency_p95 > acceptance["latency_p95_ms_max"]:
            errors.append(f"{item['name']} latency gate failed")
        if abs(item["valid_ratio"]["mean"] - valid_mean) > 1e-12:
            errors.append(f"{item['name']} valid ratio summary drifted")
        if abs(item["latency_ms"]["p95"] - latency_p95) > 1e-12:
            errors.append(f"{item['name']} latency summary drifted")
        if not item["passed"]:
            errors.append(f"{item['name']} is not marked passed")

    fixture = summary_path.parent / "runtime_fixture"
    if tree_sha256(fixture) != status["runtime"]["fixture_tree_sha256"]:
        errors.append("Phase 5-D runtime fixture tree drifted")
    manifest = fixture / "manifest.json"
    if summary["runtime_fixture_manifest_sha256"] != sha256(manifest):
        errors.append("runtime fixture manifest differs from robustness summary")


def validate_runtime(status, contract, errors):
    summary_path = check_hash(status["runtime"]["summary"], errors, "runtime summary")
    topology_path = check_hash(status["implementation"]["runtime_dataflow"], errors, "runtime dataflow")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        acceptance = contract["runtime"]["acceptance"]
        if summary["status"] != "runtime_gate_passed" or not summary["gate_passed"]:
            errors.append("Phase 5-D runtime gate is not frozen as passed")
        if summary["frames_expected"] != acceptance["exact_output_frames"] or summary["frames_received"] != acceptance["exact_output_frames"]:
            errors.append("Phase 5-D runtime frame count drifted")
        if summary["source_frame_match_ratio"] < acceptance["source_frame_match_ratio_min"]:
            errors.append("runtime source-frame pairing gate failed")
        if summary["occupancy_exact_match_ratio"] < acceptance["occupancy_exact_match_ratio_min"]:
            errors.append("runtime occupancy parity gate failed")
        if summary["valid_ratio"] < acceptance["valid_ratio_min"]:
            errors.append("runtime validity gate failed")
        if summary["latency_ms"]["p95"] > acceptance["latency_p95_ms_max"]:
            errors.append("runtime latency gate failed")
        if summary["control_output_declared"] or summary["candidate_controls_vehicle"]:
            errors.append("runtime candidate was granted control authority")
    if topology_path.is_file():
        topology = yaml.safe_load(topology_path.read_text(encoding="utf-8"))
        outputs = [output for node in topology["nodes"] for output in node.get("outputs", [])]
        if "control_cmd" in outputs:
            errors.append("Phase 5-D shadow topology declares control_cmd")
        serialized = json.dumps(topology)
        if "fast_brain_nmpc" in serialized or "control_cmd" in serialized:
            errors.append("Phase 5-D shadow topology has a control consumer or edge")


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5d_robustness_runtime_passed_control_closed":
        errors.append("Phase 5-D frozen status is invalid")
    check_hash(status["phase5c3_status"], errors, "Phase 5-C3 status")
    contract_path = check_hash(status["contract"], errors, "Phase 5-D contract")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract["promotion"]["control_promotion_allowed"]:
            errors.append("Phase 5-D contract opened control promotion")
        validate_robustness(status, contract, errors)
        validate_runtime(status, contract, errors)
    if status["control_authority"]["control_promotion_allowed"]:
        errors.append("Phase 5-D status opened control promotion")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-D multi-seed robustness and Dora runtime validation OK")
    print("3x1000 perturbed frames and 100 exact synchronized runtime frames passed")
    print("Candidate remains shadow-only; Oracle NMPC retains sole control authority")


if __name__ == "__main__":
    main()
