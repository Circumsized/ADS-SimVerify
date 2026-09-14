#!/usr/bin/env python3
import json
import os
import signal
import sys
import time
from pathlib import Path

from phase5k_faults import SafetyLedger, atomic_write_json, host_watchdog_due


def main():
    output = Path(sys.argv[1]).resolve()
    timeout_s = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "fault_receipt.json"
    ledger_path = output / "actuator.safety.ledger"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if receipt_path.is_file() and ledger_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            ledger = SafetyLedger.read(ledger_path)
            if host_watchdog_due(receipt, ledger):
                signal_sent_ns = time.monotonic_ns()
                os.kill(int(ledger["pid"]), signal.SIGUSR1)
                settle_deadline = time.monotonic() + 2.0
                while time.monotonic() < settle_deadline:
                    stopped = SafetyLedger.read(ledger_path)
                    if (
                        stopped.get("reason") == "watchdog_zero"
                        and float(stopped.get("linear", 1.0)) == 0.0
                        and float(stopped.get("angular", 1.0)) == 0.0
                    ):
                        report = {
                            "schema_version": "phase5k-host-watchdog-v1",
                            "target": "daemon",
                            "plant_pid": int(ledger["pid"]),
                            "signal": "SIGUSR1",
                            "signal_sent_monotonic_ns": signal_sent_ns,
                            "zero_monotonic_ns": int(stopped["monotonic_ns"]),
                            "signal_to_zero_ms": (int(stopped["monotonic_ns"]) - signal_sent_ns) / 1e6,
                            "kill_to_zero_ms": (
                                int(stopped["monotonic_ns"]) - int(receipt["kill_monotonic_ns"])
                            ) / 1e6,
                            "confirmed": True,
                        }
                        atomic_write_json(output / "host_watchdog_receipt.json", report)
                        print(json.dumps(report, indent=2))
                        return
                    time.sleep(0.002)
                raise SystemExit("plant did not acknowledge host watchdog within 2 seconds")
        time.sleep(0.005)
    raise SystemExit(f"daemon fault was not observed within {timeout_s:.1f} seconds")


if __name__ == "__main__":
    main()
