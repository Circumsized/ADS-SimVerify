#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase5/phase5k_status.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate_runs(runs):
    return {
        "run_modes": [run["run_mode"] for run in runs],
        "frames": sum(int(run["frames"]) for run in runs),
        "collision_count": sum(int(run["metrics"]["collision_count"]) for run in runs),
        "all_run_gates_passed": all(bool(run["gate_passed"]) for run in runs),
    }


def main():
    status = json.loads(STATUS.read_text())
    errors = []
    for name in ("phase5j_status", "contract"):
        reference = status[name]
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"{name} hash mismatch")
    for name, reference in status["implementation"].items():
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"{name} hash mismatch")
    runs = []
    for mode, reference in status["runs"].items():
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"{mode} evidence hash mismatch")
            continue
        runs.append(json.loads(path.read_text()))
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    aggregate = aggregate_runs(runs)
    contract = json.loads((ROOT / status["contract"]["path"]).read_text())
    expected_modes = ["hour_endurance", *contract["fault_gate"]["runs"]]
    if aggregate != status["aggregate"]:
        errors.append("aggregate differs from frozen status")
    if aggregate["run_modes"] != expected_modes:
        errors.append("run membership or order drifted")
    if aggregate["collision_count"] or not aggregate["all_run_gates_passed"]:
        errors.append("Phase 5-K safety invariant failed")
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("Phase 5-K infrastructure, resource-fault and hour-endurance validation OK")
    print(f"{aggregate['frames']} frames; zero collisions; all five gates passed")


if __name__ == "__main__":
    main()
