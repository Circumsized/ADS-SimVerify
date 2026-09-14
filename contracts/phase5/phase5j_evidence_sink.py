#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
from pathlib import Path


FIELDS = [
    "source_frame_id", "run_mode", "scenario_name", "episode_index",
    "sensor_frame_id", "sensor_frozen", "controller_generation",
    "safety_state", "safety_reason", "reset_applied", "command_linear_mps",
    "command_angular_radps", "x_m", "y_m", "yaw_rad", "velocity_mps",
    "sensor_to_wheel_ms", "static_collision", "dynamic_collision",
    "actuator_reason",
]


def metadata_value(event, key, default=None):
    metadata = event.get("metadata") or {}
    if key in metadata:
        return metadata[key]
    return (metadata.get("parameters") or {}).get(key, default)


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fault_stop_latency(rows, event_frame):
    for row in rows:
        frame_id = int(row["source_frame_id"])
        if frame_id < int(event_frame):
            continue
        if float(row["command_linear_mps"]) == 0.0 and float(row["command_angular_radps"]) == 0.0:
            return frame_id - int(event_frame)
    return None


def run_metrics(rows, expected_generation):
    scenario_order = []
    for row in rows:
        name = row["scenario_name"]
        if name not in scenario_order:
            scenario_order.append(name)
    reset_indices = [index for index, row in enumerate(rows) if bool(int(row["reset_applied"]))]
    last_reset = reset_indices[-1] if reset_indices else -1
    latencies = [float(row["sensor_to_wheel_ms"]) for row in rows if float(row["sensor_to_wheel_ms"]) >= 0.0]
    return {
        "exact_source_frame_ids": [int(row["source_frame_id"]) for row in rows] == list(range(len(rows))),
        "scenario_coverage": scenario_order,
        "episodes": len({int(row.get("episode_index", 0)) for row in rows}),
        "sensor_frozen_frames": sum(bool(int(row["sensor_frozen"])) for row in rows),
        "reset_events": len(reset_indices),
        "active_after_last_reset": last_reset >= 0 and any(
            row["safety_state"] == "active" and float(row["command_linear_mps"]) > 0.0
            for row in rows[last_reset + 1 :]
        ),
        "wrong_generation_commands": sum(
            float(row["command_linear_mps"]) != 0.0
            and int(row["controller_generation"]) != int(expected_generation)
            for row in rows
        ),
        "collision_count": sum(
            int(row["static_collision"]) + int(row["dynamic_collision"]) for row in rows
        ),
        "active_frames": sum(
            row["safety_state"] == "active"
            and row["actuator_reason"] == "fresh"
            and (float(row["command_linear_mps"]) != 0.0 or float(row["command_angular_radps"]) != 0.0)
            for row in rows
        ),
        "sensor_to_wheel_p95_ms": percentile(latencies, 0.95),
        "sensor_to_wheel_max_ms": max(latencies, default=0.0),
    }


def common_gate_passes(metrics, expected_scenarios, minimum_episodes, latency_p95_ms_max):
    return bool(
        metrics["exact_source_frame_ids"]
        and metrics["scenario_coverage"] == list(expected_scenarios)
        and metrics["episodes"] >= int(minimum_episodes)
        and metrics["wrong_generation_commands"] == 0
        and metrics["collision_count"] == 0
        and metrics["sensor_to_wheel_p95_ms"] <= float(latency_p95_ms_max)
    )


def main():
    from dora import Node

    root = Path(__file__).resolve().parents[2]
    contract = json.loads((root / "contracts/phase5/phase5j_contract.json").read_text())
    output = Path(os.environ["PHASE5J_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_mode = os.environ["PHASE5J_RUN_MODE"]
    generation = int(os.environ["PHASE5J_GENERATION"])
    rows = []
    events = []
    controller_pid = None
    supervisor_pid = None
    node = Node()
    complete = False
    idle_after_complete = 0
    while not complete or idle_after_complete < 3:
        event = node.next(timeout=0.5)
        if event is None:
            if complete:
                idle_after_complete += 1
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "fault_event":
            events.append({
                "frame_id": int(metadata_value(event, "source_frame_id", -1)),
                "action": str(metadata_value(event, "action", "unknown")),
                "target": str(metadata_value(event, "target", "none")),
                "pid": int(metadata_value(event, "pid", -1)),
                "confirmed": bool(metadata_value(event, "confirmed", False)),
            })
            continue
        if event["id"] == "controller_heartbeat":
            controller_pid = int(metadata_value(event, "pid", -1))
            continue
        if event["id"] == "supervisor_heartbeat":
            supervisor_pid = int(metadata_value(event, "pid", -1))
            continue
        if event["id"] == "run_complete":
            complete = True
            continue
        if event["id"] != "plant_telemetry":
            continue
        values = event["value"].to_numpy()
        rows.append({
            "source_frame_id": int(metadata_value(event, "source_frame_id")),
            "run_mode": run_mode,
            "scenario_name": str(metadata_value(event, "scenario_name")),
            "episode_index": int(metadata_value(event, "episode_index", 0)),
            "sensor_frame_id": int(metadata_value(event, "sensor_frame_id", -1)),
            "sensor_frozen": int(bool(metadata_value(event, "sensor_frozen", False))),
            "controller_generation": int(metadata_value(event, "controller_generation", -1)),
            "safety_state": str(metadata_value(event, "safety_state", "boot")),
            "safety_reason": str(metadata_value(event, "safety_reason", "none")),
            "reset_applied": int(bool(metadata_value(event, "reset_applied", False))),
            "command_linear_mps": float(values[0]),
            "command_angular_radps": float(values[1]),
            "x_m": float(values[2]), "y_m": float(values[3]),
            "yaw_rad": float(values[4]), "velocity_mps": float(values[5]),
            "sensor_to_wheel_ms": float(values[6]),
            "static_collision": int(values[7]), "dynamic_collision": int(values[8]),
            "actuator_reason": str(metadata_value(event, "actuator_reason", "none")),
        })
    rows.sort(key=lambda row: int(row["source_frame_id"]))
    telemetry = output / "frames.csv"
    with telemetry.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metrics = run_metrics(rows, generation)
    confirmed_kills = [event for event in events if event["action"] == "sigkill" and event["confirmed"]]
    frozen_frames = [int(row["source_frame_id"]) for row in rows if bool(int(row["sensor_frozen"]))]
    event_frame = confirmed_kills[0]["frame_id"] if confirmed_kills else (frozen_frames[0] if frozen_frames else None)
    stop_latency = fault_stop_latency(rows, event_frame) if event_frame is not None else 0
    metrics["confirmed_sigkills"] = len(confirmed_kills)
    metrics["maximum_fault_stop_latency_frames"] = stop_latency if stop_latency is not None else 10**9
    required_frames = contract["endurance"]["minimum_frames"] if run_mode == "endurance" else contract["fault_gate"]["frames_per_run"]
    expected_scenarios = contract["endurance"]["scenarios"] if run_mode == "endurance" else ["straight_aisle"]
    mode_passed = True
    if run_mode in {"controller_sigkill", "supervisor_sigkill"}:
        expected_target = "controller" if run_mode == "controller_sigkill" else "supervisor"
        mode_passed = (
            len(confirmed_kills) == 1
            and confirmed_kills[0]["target"] == expected_target
            and metrics["maximum_fault_stop_latency_frames"] <= contract["fault_gate"]["stop_latency_frames_max"]
        )
    elif run_mode == "sensor_freeze":
        mode_passed = (
            metrics["sensor_frozen_frames"] >= 8
            and metrics["maximum_fault_stop_latency_frames"] <= contract["fault_gate"]["stop_latency_frames_max"]
            and metrics["reset_events"] >= 1
            and metrics["active_after_last_reset"]
        )
    elif run_mode == "restart_recovery":
        mode_passed = metrics["active_frames"] >= contract["fault_gate"]["restart_active_frames_min"]
    minimum_episodes = (
        contract["endurance"]["scenario_cycles"] * len(expected_scenarios)
        if run_mode == "endurance"
        else 1
    )
    passed = bool(
        len(rows) >= required_frames
        and common_gate_passes(
            metrics,
            expected_scenarios,
            minimum_episodes,
            contract["timing"]["sensor_to_wheel_p95_ms_max"],
        )
        and mode_passed
    )
    summary = {
        "schema_version": "phase5j-run-v1",
        "status": "run_gate_passed" if passed else "run_gate_rejected",
        "run_mode": run_mode,
        "frames": len(rows),
        "controller_generation": generation,
        "controller_pid": controller_pid,
        "supervisor_pid": supervisor_pid,
        "metrics": metrics,
        "fault_events": events,
        "telemetry": telemetry.name,
        "telemetry_sha256": sha256(telemetry),
        "gate_passed": passed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
