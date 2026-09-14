#!/usr/bin/env python3
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5b_status.json"


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
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def close(actual, expected, tolerance=1e-6):
    return abs(actual - expected) <= tolerance * max(1.0, abs(expected))


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5b_shadow_frozen_candidate_rejected":
        errors.append("Phase 5-B status is not frozen as a rejected candidate")
    phase5a_path = check_hash(status["phase5a_status"], errors, "Phase 5-A status")
    if phase5a_path.is_file():
        phase5a = json.loads(phase5a_path.read_text(encoding="utf-8"))
        if phase5a["status"] != "phase5a_oracle_nmpc_frozen":
            errors.append("Phase 5-B does not descend from frozen Phase 5-A")
    for name, reference in status["implementation"].items():
        check_hash(reference, errors, name)
    model_path = check_hash(status["candidate_model"], errors, "candidate model")
    summary_path = check_hash(status["shadow_evidence"]["summary"], errors, "shadow summary")
    telemetry_path = check_hash(status["shadow_evidence"]["telemetry"], errors, "shadow telemetry")

    if summary_path.is_file() and telemetry_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with telemetry_path.open(newline="", encoding="ascii") as source:
            rows = list(csv.DictReader(source))
        if summary["status"] != "shadow_evidence_complete_candidate_rejected":
            errors.append("shadow summary did not reject the current candidate")
        authority = summary["control_authority"]
        if authority["shadow_outputs_can_control"] or authority["control_output_declared_by_this_program"]:
            errors.append("a shadow output has control authority")
        if len(rows) != summary["frame_count"] or len(rows) < summary["minimum_gate_frames"]:
            errors.append("shadow evidence does not satisfy the frozen frame count")
        frame_ids = [int(row["source_frame_id"]) for row in rows]
        if frame_ids != list(range(len(rows))):
            errors.append("shadow frame ids are not exact and contiguous")
        if summary["synchronization"]["exact_frame_ratio"] != 1.0:
            errors.append("shadow modalities are not exactly synchronized")
        if not all(int(row["candidate_valid"]) == 1 for row in rows):
            errors.append("candidate produced an invalid frame")
        if summary["candidate"]["model_sha256"] != sha256(model_path):
            errors.append("summary references a different candidate model")
        if summary["candidate"]["eligible_to_control"]:
            errors.append("rejected candidate was marked eligible to control")
        if summary["telemetry_sha256"] != sha256(telemetry_path):
            errors.append("telemetry hash differs from summary")
        for item in summary["evidence"]:
            evidence_path = summary_path.parent / item["path"]
            if not evidence_path.is_file() or sha256(evidence_path) != item["sha256"]:
                errors.append(f"evidence image is missing or has drifted: {item['path']}")

        candidate_latency = [float(row["candidate_latency_ms"]) for row in rows]
        measured_p95 = percentile(candidate_latency, 0.95)
        if not close(measured_p95, summary["latency_ms"]["pidnet_total"]["p95"]):
            errors.append("candidate latency summary differs from telemetry")
        if measured_p95 > summary["gate_thresholds"]["latency_p95_ms_max"]:
            errors.append("candidate does not meet the 20 Hz latency budget")

        for method in ("depth_gt", "pidnet_flat", "pidnet_depth"):
            counts = {
                name: sum(int(row[f"{method}_{name}_count"]) for row in rows)
                for name in ("true_positive", "true_negative", "false_positive", "false_negative")
            }
            frozen_counts = summary["methods"][method]["micro_confusion"]
            for name, value in counts.items():
                if frozen_counts[name] != value:
                    errors.append(f"{method} {name} differs from telemetry")
            if method.startswith("pidnet") and summary["methods"][method]["gate_passed"]:
                errors.append(f"rejected {method} unexpectedly passed its metric gate")

        risk = summary["risk_decision_coverage"]
        oracle_stop_frames = sum(int(row["oracle_stop"]) for row in rows)
        if risk["oracle_stop_frames"] != oracle_stop_frames:
            errors.append("Oracle stop-frame count differs from telemetry")
        if oracle_stop_frames == 0 and risk["dangerous_false_go_claim_allowed"]:
            errors.append("dangerous false-go recall was claimed without positive stop frames")
        verdict = summary["verdict"]
        if not verdict["evidence_volume_passed"] or not verdict["latency_passed"]:
            errors.append("Phase 5-B infrastructure gate did not pass")
        if verdict["candidate_gate_passed"] or verdict["candidate_control_gate"] != "closed":
            errors.append("candidate perception gate was opened")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-B shadow evidence validation OK")
    print("1143 exact frames; Oracle retained sole control authority")
    print("Latency passed; PIDNet candidate metric/control gates remain closed")


if __name__ == "__main__":
    main()
