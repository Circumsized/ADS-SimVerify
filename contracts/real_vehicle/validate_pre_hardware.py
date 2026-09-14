#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    ROOT
    / "artifacts/real_vehicle_acceptance/pre_hardware_20260716_182135/summary.json"
)
REQUIRED_GATES = (
    "topology",
    "camera",
    "localization",
    "global_planning",
    "collision_supervisor",
    "actuator",
)


def summary_errors(summary):
    errors = []
    if summary.get("status") != "real_vehicle_gate_blocked" or summary.get("gate_passed") is not False:
        errors.append("pre-hardware status must remain explicitly blocked")
    if summary.get("real_vehicle_control_allowed") is not False:
        errors.append("real-vehicle control must remain forbidden before hardware evidence")
    if set(summary.get("blocked_gates", [])) != set(REQUIRED_GATES):
        errors.append("pre-hardware evidence must explicitly block all six gates")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    errors = summary_errors(summary)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("Real-vehicle pre-hardware acceptance validation OK")
    print("all six physical gates are explicitly blocked; control remains forbidden")


if __name__ == "__main__":
    main()
