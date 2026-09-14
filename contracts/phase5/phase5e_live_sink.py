#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dora import Node

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contracts/phase5"))
from phase5b_shadow_replay import aggregate, control_roi, nearest_center_obstacle  # noqa: E402
from phase5c2_geometry_upper_bound import occupancy_metrics_contract  # noqa: E402
from phase5c_dynamic_upper_bound import STOP_DISTANCE_M, stop_metrics  # noqa: E402
from phase5d_runtime_node import metadata_value, source_frame_id  # noqa: E402

FIELDS = [
    "source_frame_id",
    "dynamic_mode",
    "oracle_stop",
    "candidate_stop",
    "runtime_valid",
    "candidate_valid_ratio",
    "latency_ms",
    "occupied_iou",
    "free_iou",
    "false_free_rate",
    "false_occupied_rate",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    output = Path(os.environ["PHASE5E_LIVE_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((ROOT / "contracts/phase5/phase5e_contract.json").read_text())
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    formal_frame_count = contract["live_isaac_shadow"]["frames"]
    frame_count = int(os.environ.get("PHASE5E_MAX_FRAMES", formal_frame_count))
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    streams = {
        "oracle_bev": {},
        "oracle_valid": {},
        "depth_reference_valid": {},
        "shadow_bev_grid": {},
        "shadow_valid": {},
    }
    oracle_metadata = {}
    candidate_metadata = {}
    rows = []
    node = Node()
    telemetry_path = output / "frames.csv"
    with telemetry_path.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        while len(rows) < frame_count:
            event = node.next(timeout=2.0)
            if event is None:
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] not in streams:
                continue
            frame_id = source_frame_id(event)
            if frame_id is None:
                continue
            stream = event["id"]
            streams[stream][frame_id] = event["value"].to_numpy().copy()
            if stream == "oracle_bev":
                oracle_metadata[frame_id] = {
                    "dynamic_mode": str(metadata_value(event, "dynamic_mode")),
                    "oracle_stop": bool(metadata_value(event, "oracle_stop")),
                }
            elif stream == "shadow_bev_grid":
                candidate_metadata[frame_id] = {
                    "valid": bool(metadata_value(event, "valid")),
                    "latency_ms": float(metadata_value(event, "latency_ms") or 0.0),
                }
            if not all(frame_id in values for values in streams.values()):
                continue
            oracle = streams["oracle_bev"].pop(frame_id).reshape(192, 192) > 0
            oracle_valid = streams["oracle_valid"].pop(frame_id).reshape(192, 192) > 0
            depth_reference = streams["depth_reference_valid"].pop(frame_id).reshape(192, 192) > 0
            candidate = streams["shadow_bev_grid"].pop(frame_id).reshape(192, 192) > 0
            candidate_valid = streams["shadow_valid"].pop(frame_id).reshape(192, 192) > 0
            reference_roi = fixed_roi & oracle_valid & depth_reference
            common_valid = reference_roi & candidate_valid
            metrics = occupancy_metrics_contract(oracle, candidate, common_valid)
            nearest = nearest_center_obstacle(candidate, fixed_roi & candidate_valid, bev)
            candidate_stop = bool(nearest is not None and nearest <= STOP_DISTANCE_M)
            metadata = oracle_metadata.pop(frame_id)
            runtime = candidate_metadata.pop(frame_id)
            row = {
                "source_frame_id": frame_id,
                "dynamic_mode": metadata["dynamic_mode"],
                "oracle_stop": int(metadata["oracle_stop"]),
                "candidate_stop": int(candidate_stop),
                "runtime_valid": int(runtime["valid"]),
                "candidate_valid_ratio": float(common_valid.sum() / max(reference_roi.sum(), 1)),
                "latency_ms": runtime["latency_ms"],
                **{
                    name: metrics[name]
                    for name in (
                        "occupied_iou",
                        "free_iou",
                        "false_free_rate",
                        "false_occupied_rate",
                    )
                },
            }
            writer.writerow(row)
            rows.append(row)
            if len(rows) % 100 == 0:
                print(f"[Phase 5-E live sink] frames={len(rows)}/{frame_count}")

    rows.sort(key=lambda row: int(row["source_frame_id"]))
    acceptance = contract["live_isaac_shadow"]["acceptance"]
    metrics = {
        name: aggregate(rows, name)
        for name in (
            "occupied_iou",
            "free_iou",
            "false_free_rate",
            "false_occupied_rate",
        )
    }
    stopping = stop_metrics(
        [
            {
                "oracle_stop": bool(row["oracle_stop"]),
                "depth_gt_stop": bool(row["candidate_stop"]),
            }
            for row in rows
        ]
    )
    valid_ratio = aggregate(rows, "candidate_valid_ratio")
    latency = aggregate(rows, "latency_ms")
    exact_ids = [int(row["source_frame_id"]) for row in rows] == list(range(frame_count))
    mode_counts = Counter(row["dynamic_mode"] for row in rows)
    gate_passed = bool(
        frame_count == formal_frame_count
        and exact_ids
        and len(rows) == frame_count
        and metrics["occupied_iou"]["mean"] >= acceptance["occupied_iou_mean_min"]
        and metrics["free_iou"]["mean"] >= acceptance["free_iou_mean_min"]
        and metrics["false_free_rate"]["mean"] <= acceptance["false_free_rate_mean_max"]
        and metrics["false_occupied_rate"]["mean"] <= acceptance["false_occupied_rate_mean_max"]
        and stopping["stop_recall"] >= acceptance["stop_recall_min"]
        and stopping["go_specificity"] >= acceptance["go_specificity_min"]
        and valid_ratio["mean"] >= acceptance["candidate_valid_ratio_mean_min"]
        and latency["p95"] <= acceptance["latency_p95_ms_max"]
        and all(bool(row["runtime_valid"]) for row in rows)
    )
    summary = {
        "schema_version": "phase5e-live-shadow-v1",
        "status": (
            "live_shadow_passed"
            if gate_passed
            else "smoke_only"
            if frame_count < formal_frame_count
            else "live_shadow_rejected"
        ),
        "frames": len(rows),
        "exact_source_frame_ids": exact_ids,
        "dynamic_modes": dict(mode_counts),
        "metrics": metrics,
        "stop_decision": stopping,
        "candidate_valid_ratio": valid_ratio,
        "latency_ms": latency,
        "runtime_valid_ratio": sum(bool(row["runtime_valid"]) for row in rows) / max(len(rows), 1),
        "telemetry": telemetry_path.name,
        "telemetry_sha256": sha256(telemetry_path),
        "gate_passed": gate_passed,
        "control_output_declared": False,
        "candidate_controls_vehicle": False,
        "control_promotion_allowed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"Phase 5-E live artifacts: {output}")


if __name__ == "__main__":
    main()
