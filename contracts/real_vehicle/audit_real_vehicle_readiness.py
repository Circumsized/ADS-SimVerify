#!/usr/bin/env python3
import argparse
import copy
import glob
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from real_vehicle_acceptance import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    audit_topology_text,
    evaluate_acceptance,
)


def build_preflight_evidence(topology_text, camera_devices, supplied):
    evidence = copy.deepcopy(supplied)
    topology = evidence.setdefault("topology", {})
    topology.update(audit_topology_text(topology_text))

    camera = evidence.setdefault("camera", {})
    camera["detected_devices"] = list(camera_devices)
    camera["device_present"] = bool(camera_devices)
    if not camera_devices:
        camera["calibration_gate_passed"] = False
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, default=ROOT / "dora_dataflow.yaml")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.topology.is_file():
        raise SystemExit(f"topology does not exist: {args.topology}")
    supplied = {}
    if args.evidence:
        if not args.evidence.is_file():
            raise SystemExit(f"evidence does not exist: {args.evidence}")
        supplied = json.loads(args.evidence.read_text(encoding="utf-8"))
    devices = sorted(glob.glob("/dev/video*"))
    evidence = build_preflight_evidence(
        args.topology.read_text(encoding="utf-8"), devices, supplied
    )
    result = evaluate_acceptance(evidence, DEFAULT_THRESHOLDS)
    output = args.output or ROOT / "artifacts/real_vehicle_acceptance" / time.strftime(
        "pre_hardware_%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": "real-vehicle-readiness-audit-v1",
        "status": "real_vehicle_gate_passed" if result["gate_passed"] else "real_vehicle_gate_blocked",
        "topology": str(args.topology.resolve()),
        "supplied_evidence": str(args.evidence.resolve()) if args.evidence else None,
        "evidence": evidence,
        **result,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"Real-vehicle readiness evidence: {path}")


if __name__ == "__main__":
    main()
