#!/usr/bin/env python3
import csv
import json
import os
import time
from pathlib import Path

from phase5j_evidence_sink import FIELDS, fault_stop_latency, metadata_value, run_metrics, sha256
from phase5k_evidence import coordinator_gate, hour_gate, recovery_gate, resource_receipt_gate
from phase5k_faults import SafetyLedger, atomic_write_json


def telemetry_row(event, run_mode):
    values = event["value"].to_numpy()
    return {
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
        "x_m": float(values[2]),
        "y_m": float(values[3]),
        "yaw_rad": float(values[4]),
        "velocity_mps": float(values[5]),
        "sensor_to_wheel_ms": float(values[6]),
        "static_collision": int(values[7]),
        "dynamic_collision": int(values[8]),
        "actuator_reason": str(metadata_value(event, "actuator_reason", "none")),
    }


def load_receipt(output, run_mode):
    names = {
        "coordinator_sigkill": "fault_receipt.json",
        "gpu_oom_recovery": "gpu_oom_receipt.json",
        "disk_full_recovery": "disk_full_receipt.json",
    }
    name = names.get(run_mode)
    path = output / name if name else None
    return json.loads(path.read_text()) if path is not None and path.is_file() else {}


def main():
    from dora import Node

    root = Path(__file__).resolve().parents[2]
    contract = json.loads((root / "contracts/phase5/phase5k_contract.json").read_text())
    output = Path(os.environ["PHASE5K_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_mode = os.environ["PHASE5K_RUN_MODE"]
    generation = int(os.environ["PHASE5K_GENERATION"])
    required_frames = int(os.environ["PHASE5K_MAX_FRAMES"])
    telemetry = output / "frames.csv"
    rows = []
    events = []
    controller_pid = None
    supervisor_pid = None
    complete = False
    idle_after_complete = 0
    started_ns = time.monotonic_ns()
    completed_ns = None
    node = Node()

    with telemetry.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        target.flush()
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
                completed_ns = time.monotonic_ns()
                continue
            if event["id"] != "plant_telemetry":
                continue
            row = telemetry_row(event, run_mode)
            rows.append(row)
            writer.writerow(row)
            target.flush()

    rows.sort(key=lambda row: int(row["source_frame_id"]))
    metrics = run_metrics(rows, generation)
    metrics["frames"] = len(rows)
    metrics["active_ratio"] = metrics["active_frames"] / len(rows) if rows else 0.0
    metrics["wall_duration_s"] = ((completed_ns or time.monotonic_ns()) - started_ns) / 1e9
    receipt = load_receipt(output, run_mode)
    injection_frame = int(receipt.get("source_frame_id", contract["fault_gate"]["injection_frame"]))
    stop_latency = fault_stop_latency(rows, injection_frame) if run_mode != "hour_endurance" else 0
    metrics["maximum_fault_stop_latency_frames"] = stop_latency if stop_latency is not None else 10**9
    metrics["fault_observed"] = bool(receipt.get("observed", receipt.get("confirmed", False)))
    latency_limit = contract["timing"]["sensor_to_wheel_p95_ms_max"]
    common = bool(
        len(rows) == required_frames
        and metrics["exact_source_frame_ids"]
        and metrics["wrong_generation_commands"] == 0
        and metrics["collision_count"] == 0
        and metrics["sensor_to_wheel_p95_ms"] <= latency_limit
    )
    if run_mode == "hour_endurance":
        passed = hour_gate(
            metrics,
            contract["hour_endurance"]["frames"],
            contract["hour_endurance"]["duration_s"],
            contract["hour_endurance"]["active_ratio_min"],
            latency_limit,
        )
    elif run_mode == "coordinator_sigkill":
        ledger_path = output / "actuator.safety.ledger"
        ledger = SafetyLedger.read(ledger_path) if ledger_path.is_file() else {}
        passed = coordinator_gate(
            metrics,
            receipt,
            ledger,
            contract["fault_gate"]["out_of_band_stop_latency_ms_max"],
        )
    elif run_mode in {"gpu_oom_recovery", "disk_full_recovery"}:
        passed = bool(
            common
            and resource_receipt_gate(run_mode, receipt)
            and recovery_gate(metrics, contract["fault_gate"]["stop_latency_frames_max"])
        )
    else:
        passed = False
    summary = {
        "schema_version": "phase5k-run-v1",
        "status": "run_gate_passed" if passed else "run_gate_rejected",
        "run_mode": run_mode,
        "frames": len(rows),
        "controller_generation": generation,
        "controller_pid": controller_pid,
        "supervisor_pid": supervisor_pid,
        "metrics": metrics,
        "fault_events": events,
        "fault_receipt": receipt,
        "telemetry": telemetry.name,
        "telemetry_sha256": sha256(telemetry),
        "gate_passed": passed,
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
