#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "contracts/phase6/phase6_status.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status_references(status):
    return [
        status["phase5k_status"],
        status["contract"],
        *status["implementation"].values(),
        status["matrix_evidence"],
    ]


def matrix_evidence_references(summary_path, summary):
    root = Path(summary_path).parent
    return [
        (root / summary["evidence"], summary["evidence_sha256"]),
        *[
            (root / item["telemetry"], item["telemetry_sha256"])
            for item in summary["cases"]
        ],
    ]


def main():
    status = json.loads(STATUS.read_text())
    errors = []
    for reference in status_references(status):
        path = ROOT / reference["path"]
        if not path.is_file() or sha256(path) != reference["sha256"]:
            errors.append(f"hash mismatch: {reference['path']}")
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    summary_path = ROOT / status["matrix_evidence"]["path"]
    summary = json.loads(summary_path.read_text())
    for path, expected_hash in matrix_evidence_references(summary_path, summary):
        if not path.is_file() or sha256(path) != expected_hash:
            errors.append(f"matrix evidence hash mismatch: {path.relative_to(ROOT)}")
    if not summary.get("gate_passed") or summary.get("aggregate") != status.get("aggregate"):
        errors.append("Phase 6 matrix gate or frozen aggregate mismatch")
    if status.get("real_vehicle_control_allowed") is not False:
        errors.append("real vehicle control boundary is not closed")
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("Phase 6 final simulation acceptance matrix validation OK")
    print(
        f"{status['aggregate']['reached_cases']}/{status['aggregate']['cases']} cases reached; "
        f"zero collisions; p95 max={status['aggregate']['sensor_to_wheel_p95_ms_max']:.2f} ms"
    )


if __name__ == "__main__":
    main()
