#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
from pathlib import Path

FIELDS = [
    "source_frame_id", "run_mode", "safety_state", "safety_reason",
    "reset_applied", "command_linear_mps", "command_angular_radps",
    "x_m", "y_m", "yaw_rad", "velocity_mps", "render_ms",
    "inference_ms", "depth_lift_ms", "nmpc_ms", "sensor_to_wheel_ms",
    "static_collision", "dynamic_collision", "actuator_reason",
]


def metadata_value(event, key):
    metadata = event.get("metadata") or {}
    if key in metadata:
        return metadata[key]
    return (metadata.get("parameters") or {}).get(key)


def source_frame_id(event):
    try:
        return int(metadata_value(event, "source_frame_id"))
    except (TypeError, ValueError):
        return None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(values):
    values = [float(value) for value in values]
    return {
        "mean": sum(values) / max(len(values), 1),
        "p95": percentile(values, 0.95),
        "max": max(values, default=0.0),
    }


def run_metrics(rows):
    latencies = [row["sensor_to_wheel_ms"] for row in rows]
    jitter = [abs(float(current) - float(previous)) for previous, current in zip(latencies, latencies[1:])]
    startup_zero = 0
    for row in rows:
        if row["safety_state"] == "active":
            break
        startup_zero += int(float(row["command_linear_mps"]) == 0.0 and float(row["command_angular_radps"]) == 0.0)
    reset_indices = [index for index, row in enumerate(rows) if bool(row["reset_applied"])]
    last_reset = reset_indices[-1] if reset_indices else -1
    supervisor_watchdog = sum(
        row["safety_reason"] in {"stale_command", "stale_health", "missing_runtime_input"}
        for row in rows
    )
    actuator_watchdog = sum(row["actuator_reason"] == "actuator_watchdog_timeout" for row in rows)
    return {
        "exact_source_frame_ids": [int(row["source_frame_id"]) for row in rows] == list(range(len(rows))),
        "startup_zero_frames": startup_zero,
        "active_frames": sum(row["safety_state"] == "active" for row in rows),
        "active_command_ratio": sum(row["safety_state"] == "active" for row in rows) / max(len(rows), 1),
        "emergency_stop_frames": sum(row["safety_state"] == "emergency_stop" for row in rows),
        "supervisor_watchdog_frames": supervisor_watchdog,
        "actuator_watchdog_frames": actuator_watchdog,
        "watchdog_stop_frames": supervisor_watchdog + actuator_watchdog,
        "reset_events": len(reset_indices),
        "active_after_last_reset": last_reset >= 0 and any(
            row["safety_state"] == "active" for row in rows[last_reset + 1 :]
        ),
        "deadline_miss_ratio": sum(float(value) > 50.0 for value in latencies) / max(len(rows), 1),
        "sensor_to_wheel_ms": aggregate(latencies),
        "sensor_to_wheel_jitter_ms": aggregate(jitter),
        "static_collision_count": sum(int(row["static_collision"]) for row in rows),
        "dynamic_collision_count": sum(int(row["dynamic_collision"]) for row in rows),
    }


def gate_status(formal, passed):
    if not formal:
        return "smoke_only"
    return "run_gate_passed" if passed else "run_gate_rejected"


def localization_sequence_valid(frame_ids, decoded_valid, frame_count, control_rate_hz=20, localization_rate_hz=2):
    period = control_rate_hz // localization_rate_hz
    expected = list(range(0, frame_count, period))
    return list(frame_ids) == expected and len(decoded_valid) == len(expected) and all(decoded_valid)


def main():
    import cv2
    import numpy as np
    from dora import Node

    root = Path(__file__).resolve().parents[2]
    contract = json.loads((root / "contracts/phase5/phase5i_contract.json").read_text())
    acceptance = contract["multirun_gate"]
    output = Path(os.environ["PHASE5I_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_mode = os.environ["PHASE5I_RUN_MODE"]
    expected_frames = int(os.environ.get("PHASE5I_MAX_FRAMES", acceptance["frames_per_run"]))
    rows = []
    localization_frame_ids = []
    localization_decoded_valid = []
    node = Node()
    while len(rows) < expected_frames:
        event = node.next(timeout=2.0)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "localization_image_640":
            frame_id = source_frame_id(event)
            encoded = event["value"].to_numpy().astype(np.uint8, copy=False)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            localization_frame_ids.append(frame_id)
            localization_decoded_valid.append(
                decoded is not None and decoded.shape == (480, 640, 3)
            )
            continue
        if event["id"] != "articulation_telemetry":
            continue
        values = event["value"].to_numpy()
        rows.append({
            "source_frame_id": source_frame_id(event),
            "run_mode": run_mode,
            "safety_state": str(metadata_value(event, "safety_state")),
            "safety_reason": str(metadata_value(event, "safety_reason")),
            "reset_applied": int(bool(metadata_value(event, "reset_applied"))),
            "command_linear_mps": float(values[0]),
            "command_angular_radps": float(values[1]),
            "x_m": float(values[2]), "y_m": float(values[3]),
            "yaw_rad": float(values[4]), "velocity_mps": float(values[5]),
            "render_ms": float(values[6]), "inference_ms": float(values[7]),
            "depth_lift_ms": float(values[8]), "nmpc_ms": float(values[9]),
            "sensor_to_wheel_ms": float(values[10]),
            "static_collision": int(values[11]), "dynamic_collision": int(values[12]),
            "actuator_reason": str(metadata_value(event, "actuator_reason")),
        })
    rows.sort(key=lambda row: int(row["source_frame_id"]))
    telemetry = output / "frames.csv"
    with telemetry.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metrics = run_metrics(rows)
    localization_valid = localization_sequence_valid(
        localization_frame_ids,
        localization_decoded_valid,
        expected_frames,
        contract["timing"]["control_rate_hz"],
        contract["sensor_pipeline"]["xfeat_rate_hz"],
    )
    mode_gate = True
    if run_mode == "emergency_stop_reset":
        mode_gate = (
            metrics["emergency_stop_frames"] >= acceptance["emergency_stop_events_min"]
            and metrics["reset_events"] >= 1
            and metrics["active_after_last_reset"]
        )
    elif run_mode == "watchdog_reset":
        mode_gate = (
            metrics["watchdog_stop_frames"] >= acceptance["watchdog_stop_events_min"]
            and metrics["reset_events"] >= 1
            and metrics["active_after_last_reset"]
        )
    passed = bool(
        len(rows) == expected_frames
        and metrics["exact_source_frame_ids"]
        and metrics["startup_zero_frames"] >= acceptance["startup_zero_frames_min"]
        and metrics["active_command_ratio"] >= acceptance["active_command_ratio_min"]
        and metrics["deadline_miss_ratio"] <= acceptance["deadline_miss_ratio_max"]
        and metrics["sensor_to_wheel_ms"]["p95"] <= acceptance["sensor_to_wheel_p95_ms_max"]
        and metrics["sensor_to_wheel_jitter_ms"]["p95"] <= acceptance["sensor_to_wheel_jitter_p95_ms_max"]
        and metrics["static_collision_count"] == acceptance["static_collision_count"]
        and metrics["dynamic_collision_count"] == acceptance["dynamic_collision_count"]
        and localization_valid
        and mode_gate
    )
    formal = expected_frames == acceptance["frames_per_run"]
    summary = {
        "schema_version": "phase5i-dora-run-v1",
        "status": gate_status(formal, passed),
        "run_mode": run_mode,
        "frames": len(rows),
        "metrics": metrics,
        "localization_image_640": {
            "frames": len(localization_frame_ids),
            "source_frame_ids": localization_frame_ids,
            "decoded_shape": [480, 640, 3],
            "exact_two_hz_native_sequence": localization_valid,
        },
        "telemetry": telemetry.name,
        "telemetry_sha256": sha256(telemetry),
        "gate_passed": passed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"Phase 5-I run artifacts: {output}")
    if formal and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
