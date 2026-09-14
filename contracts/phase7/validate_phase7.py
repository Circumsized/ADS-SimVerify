#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase7/phase7_status.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status_references(status):
    return [
        status["phase6_status"],
        status["contract"],
        *status["implementation"].values(),
        status["profile_evidence"],
        status["deployment_report"],
    ]


def recommendation_matches(profile, frozen_recommendation):
    return profile.get("recommendation", {}).get("selected") == frozen_recommendation


def main():
    status = json.loads(STATUS.read_text())
    errors = []
    for reference in status_references(status):
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"hash mismatch: {reference['path']}")
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    profile = json.loads((ROOT / status["profile_evidence"]["path"]).read_text())
    if not profile.get("gate_passed") or not recommendation_matches(
        profile, status.get("recommendation")
    ):
        errors.append("profile gate or frozen recommendation mismatch")
    if status.get("real_vehicle_control_allowed") is not False:
        errors.append("real vehicle control boundary is not closed")
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("Phase 7 deployment performance profile validation OK")
    print(
        f"recommended={status['recommendation']['name']}; "
        f"control p95={status['measured_profile']['control_pipeline_p95_ms']:.2f} ms"
    )


if __name__ == "__main__":
    main()
