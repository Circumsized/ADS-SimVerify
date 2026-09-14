#!/usr/bin/env python3
import csv
import json
import sys
import time
from pathlib import Path

from phase5j_evidence_sink import run_metrics, sha256
from phase5k_evidence import coordinator_gate, daemon_gate
from phase5k_faults import SafetyLedger, atomic_write_json


def main():
    output = Path(sys.argv[1]).resolve()
    generation = int(sys.argv[2]) if len(sys.argv) > 2 else 41
    run_mode = sys.argv[3] if len(sys.argv) > 3 else "daemon_sigkill"
    if run_mode not in {"coordinator_sigkill", "daemon_sigkill"}:
        raise SystemExit(f"unsupported process fault mode: {run_mode}")
    contract = json.loads(Path(__file__).with_name("phase5k_contract.json").read_text())
    receipt_path = output / "fault_receipt.json"
    ledger_path = output / "actuator.safety.ledger"
    telemetry = output / "frames.csv"
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if receipt_path.is_file() and ledger_path.is_file() and telemetry.is_file():
            receipt = json.loads(receipt_path.read_text())
            ledger = SafetyLedger.read(ledger_path)
            if receipt.get("confirmed") and ledger.get("reason") in {
                "coordinator_fail_safe_zero", "watchdog_zero", "terminal_zero"
            }:
                break
        time.sleep(0.05)
    else:
        raise SystemExit("daemon out-of-band evidence did not settle within 15 seconds")

    with telemetry.open(newline="", encoding="ascii") as source:
        rows = list(csv.DictReader(source))
    metrics = run_metrics(rows, generation)
    metrics["frames"] = len(rows)
    metrics["out_of_band_stop_latency_ms"] = (
        int(ledger["monotonic_ns"]) - int(receipt["kill_monotonic_ns"])
    ) / 1e6
    host_watchdog_path = output / "host_watchdog_receipt.json"
    host_watchdog = json.loads(host_watchdog_path.read_text()) if host_watchdog_path.is_file() else None
    gate = coordinator_gate if run_mode == "coordinator_sigkill" else daemon_gate
    passed = gate(metrics, receipt, ledger, contract["fault_gate"]["out_of_band_stop_latency_ms_max"])
    summary = {
        "schema_version": "phase5k-run-v1",
        "status": "run_gate_passed" if passed else "run_gate_rejected",
        "run_mode": run_mode,
        "frames": len(rows),
        "controller_generation": generation,
        "metrics": metrics,
        "fault_receipt": receipt,
        "actuator_ledger": ledger,
        "host_watchdog_receipt": host_watchdog,
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
