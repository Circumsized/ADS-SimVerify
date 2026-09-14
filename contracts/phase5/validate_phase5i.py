#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5i_status.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_runs(runs):
    metrics = [run["metrics"] for run in runs]
    return {
        "run_modes": [run["run_mode"] for run in runs],
        "frames": sum(int(run["frames"]) for run in runs),
        "maximum_sensor_to_wheel_p95_ms": max(item["sensor_to_wheel_ms"]["p95"] for item in metrics),
        "maximum_jitter_p95_ms": max(item["sensor_to_wheel_jitter_ms"]["p95"] for item in metrics),
        "emergency_stop_frames": sum(int(item["emergency_stop_frames"]) for item in metrics),
        "watchdog_stop_frames": sum(int(item["watchdog_stop_frames"]) for item in metrics),
        "reset_events": sum(int(item["reset_events"]) for item in metrics),
        "static_collision_count": sum(int(item["static_collision_count"]) for item in metrics),
        "dynamic_collision_count": sum(int(item["dynamic_collision_count"]) for item in metrics),
        "all_run_gates_passed": all(bool(run["gate_passed"]) for run in runs),
    }


def checked_reference(reference, errors, label):
    path = ROOT / reference["path"]
    if not path.is_file():
        errors.append(f"{label} is missing: {path}")
    elif sha256(path) != reference["sha256"]:
        errors.append(f"{label} hash mismatch: {path}")
    return path


def main():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []
    if status["status"] != "phase5i_formal_dora_multirun_gate_passed":
        errors.append("Phase 5-I frozen status is invalid")
    parent_path = checked_reference(status["phase5h_status"], errors, "Phase 5-H status")
    contract_path = checked_reference(status["contract"], errors, "Phase 5-I contract")
    checked_reference(status["rejected_contract"], errors, "Phase 5-I rejected v1 contract")
    for name, reference in status["implementation"].items():
        checked_reference(reference, errors, name)
    run_summaries = []
    for mode, reference in status["runs"].items():
        summary_path = checked_reference(reference["summary"], errors, f"{mode} summary")
        telemetry_path = checked_reference(reference["telemetry"], errors, f"{mode} telemetry")
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run_summaries.append(summary)
            if summary.get("telemetry_sha256") != sha256(telemetry_path):
                errors.append(f"{mode} summary telemetry hash mismatch")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    acceptance = contract["multirun_gate"]
    if parent["status"] != "phase5h_isaac_articulation_passed_real_control_closed":
        errors.append("Phase 5-H prerequisite is not passed")
    expected_modes = acceptance["runs"]
    if [run["run_mode"] for run in run_summaries] != expected_modes:
        errors.append("Phase 5-I run membership or order drifted")
    for run in run_summaries:
        metrics = run["metrics"]
        if run["frames"] != acceptance["frames_per_run"] or not run["gate_passed"]:
            errors.append(f"run gate is not passed: {run['run_mode']}")
        if not run["localization_image_640"]["exact_two_hz_native_sequence"]:
            errors.append(f"native 640x480 localization sequence failed: {run['run_mode']}")
        if metrics["sensor_to_wheel_ms"]["p95"] > acceptance["sensor_to_wheel_p95_ms_max"]:
            errors.append(f"sensor-to-wheel p95 exceeded: {run['run_mode']}")
        if metrics["sensor_to_wheel_jitter_ms"]["p95"] > acceptance["sensor_to_wheel_jitter_p95_ms_max"]:
            errors.append(f"latency jitter p95 exceeded: {run['run_mode']}")
        if metrics["deadline_miss_ratio"] > acceptance["deadline_miss_ratio_max"]:
            errors.append(f"deadline miss ratio exceeded: {run['run_mode']}")
    aggregate = aggregate_runs(run_summaries)
    if aggregate != status["aggregate"]:
        errors.append("frozen aggregate differs from run summaries")
    if aggregate["frames"] != acceptance["frames_per_run"] * len(expected_modes):
        errors.append("aggregate frame count is invalid")
    if aggregate["emergency_stop_frames"] < acceptance["emergency_stop_events_min"]:
        errors.append("emergency-stop evidence is missing")
    if aggregate["watchdog_stop_frames"] < acceptance["watchdog_stop_events_min"]:
        errors.append("watchdog evidence is missing")
    if aggregate["reset_events"] < acceptance["successful_resets_min"]:
        errors.append("reset recovery evidence is missing")
    if aggregate["static_collision_count"] or aggregate["dynamic_collision_count"]:
        errors.append("collision count is nonzero")
    if not aggregate["all_run_gates_passed"]:
        errors.append("not all run gates passed")
    if not status["control_authority"]["supervisor_is_only_wheel_command_authority"]:
        errors.append("safety supervisor is not the sole command authority")
    if status["control_authority"]["real_vehicle_control_allowed"]:
        errors.append("real vehicle control was incorrectly promoted")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 5-I formal Dora multirun validation OK")
    print(
        f"{aggregate['frames']} frames; p95 max={aggregate['maximum_sensor_to_wheel_p95_ms']:.2f} ms; "
        f"jitter p95 max={aggregate['maximum_jitter_p95_ms']:.2f} ms"
    )
    print("Watchdog, latched emergency stop, startup ordering and native 640x480 XFeat feed passed")


if __name__ == "__main__":
    main()
