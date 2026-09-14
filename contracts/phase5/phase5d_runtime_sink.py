#!/usr/bin/env python3
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from dora import Node


def metadata_value(event, key):
    metadata = event.get("metadata") or {}
    if key in metadata:
        return metadata[key]
    return (metadata.get("parameters") or {}).get(key)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    fixture = Path(os.environ["PHASE5D_FIXTURE"]).resolve()
    output = Path(os.environ["PHASE5D_RUNTIME_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture_manifest = json.loads((fixture / "manifest.json").read_text())
    expected = {int(frame["source_frame_id"]): fixture / frame["expected"] for frame in fixture_manifest["frames"]}
    grids, health = {}, {}
    node = Node()
    while len(grids) < len(expected) or len(health) < len(expected):
        event = node.next(timeout=2.0)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        try:
            frame_id = int(metadata_value(event, "source_frame_id"))
        except (TypeError, ValueError):
            continue
        data = event["value"].to_numpy().copy()
        if event["id"] == "shadow_bev_grid":
            grids[frame_id] = data
        elif event["id"] == "shadow_health":
            health[frame_id] = {
                "valid": bool(metadata_value(event, "valid")),
                "latency_ms": float(metadata_value(event, "latency_ms") or 0.0),
                "candidate_controls_vehicle": bool(metadata_value(event, "candidate_controls_vehicle")),
            }

    frame_ids = sorted(set(grids) & set(health))
    exact = []
    valid = []
    latencies = []
    for frame_id in frame_ids:
        expected_grid = np.load(expected[frame_id]).ravel()
        exact.append(bool(np.array_equal(grids[frame_id], expected_grid)))
        valid.append(health[frame_id]["valid"])
        latencies.append(health[frame_id]["latency_ms"])
    exact_ratio = sum(exact) / max(len(expected), 1)
    valid_ratio = sum(valid) / max(len(expected), 1)
    p95 = percentile(latencies, 0.95) if latencies else float("inf")
    passed = bool(
        frame_ids == list(range(len(expected)))
        and exact_ratio == 1.0
        and valid_ratio == 1.0
        and p95 <= 50.0
        and not any(item["candidate_controls_vehicle"] for item in health.values())
    )
    summary = {
        "schema_version": "phase5d-runtime-gate-v1",
        "status": "runtime_gate_passed" if passed else "runtime_gate_rejected",
        "frames_expected": len(expected),
        "frames_received": len(frame_ids),
        "source_frame_match_ratio": len(frame_ids) / max(len(expected), 1),
        "occupancy_exact_match_ratio": exact_ratio,
        "valid_ratio": valid_ratio,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p95": p95,
            "max": max(latencies) if latencies else None,
        },
        "control_output_declared": False,
        "candidate_controls_vehicle": False,
        "gate_passed": passed,
        "fixture_manifest": str(fixture / "manifest.json"),
        "fixture_manifest_sha256": sha256(fixture / "manifest.json"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"Phase 5-D runtime artifacts: {output}")


if __name__ == "__main__":
    main()
