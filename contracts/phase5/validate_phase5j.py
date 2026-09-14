#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5j_status.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_runs(runs):
    modes = [run["run_mode"] for run in runs]
    killed = next((run for run in runs if run["run_mode"] == "controller_sigkill"), None)
    restarted = next((run for run in runs if run["run_mode"] == "restart_recovery"), None)
    return {
        "run_modes": modes,
        "frames": sum(int(run["frames"]) for run in runs),
        "collision_count": sum(int(run["metrics"]["collision_count"]) for run in runs),
        "maximum_fault_stop_latency_frames": max(
            int(run["metrics"]["maximum_fault_stop_latency_frames"]) for run in runs
        ),
        "wrong_generation_commands": sum(
            int(run["metrics"]["wrong_generation_commands"]) for run in runs
        ),
        "all_run_gates_passed": all(bool(run["gate_passed"]) for run in runs),
        "restart_used_new_controller_process": bool(
            killed and restarted
            and killed["controller_pid"] != restarted["controller_pid"]
            and killed["controller_generation"] != restarted["controller_generation"]
        ),
    }


def telemetry_reference_valid(summary_path, summary):
    telemetry = Path(summary_path).parent / summary["telemetry"]
    return telemetry.is_file() and sha256(telemetry) == summary["telemetry_sha256"]


def main():
    status = json.loads(STATUS.read_text())
    errors = []
    for label, reference in [("parent", status["phase5i_status"]), ("contract", status["contract"])]:
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"{label} reference mismatch")
    for label, reference in status["implementation"].items():
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"{label} reference mismatch")
    runs = []
    for mode, reference in status["runs"].items():
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"{mode} summary mismatch")
            continue
        run = json.loads(path.read_text())
        if not telemetry_reference_valid(path, run):
            errors.append(f"{mode} telemetry mismatch")
        runs.append(run)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    aggregate = aggregate_runs(runs)
    contract = json.loads((ROOT / status["contract"]["path"]).read_text())
    expected_modes = ["endurance", *contract["fault_gate"]["runs"]]
    if aggregate != status["aggregate"]:
        errors.append("aggregate differs from frozen status")
    if aggregate["run_modes"] != expected_modes:
        errors.append("run membership or order drifted")
    if not aggregate["all_run_gates_passed"]:
        errors.append("a run gate is not passed")
    if not aggregate["restart_used_new_controller_process"]:
        errors.append("restart did not use a fresh controller process generation")
    if aggregate["collision_count"] or aggregate["wrong_generation_commands"]:
        errors.append("safety invariant failed")
    if aggregate["maximum_fault_stop_latency_frames"] > contract["fault_gate"]["stop_latency_frames_max"]:
        errors.append("fault stop latency exceeded")
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("Phase 5-J Dora endurance and fault-injection validation OK")
    print(f"{aggregate['frames']} frames; zero collisions and wrong-generation commands")
    print("Controller/supervisor SIGKILL, sensor freeze and fresh-process restart gates passed")


if __name__ == "__main__":
    main()
