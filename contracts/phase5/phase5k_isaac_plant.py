#!/usr/bin/env python3
import json
import os
import signal
import time
from pathlib import Path

import phase5j_isaac_plant as plant
from phase5k_faults import SafetyLedger, phase5j_run_mode


def main():
    run_mode = os.environ["PHASE5K_RUN_MODE"]
    os.environ["PHASE5J_RUN_MODE"] = phase5j_run_mode(run_mode)
    output = Path(os.environ["PHASE5K_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = SafetyLedger(output / "actuator.safety.ledger")
    original = plant.set_wheel_command
    sequence_id = 0
    frozen_after_process_stop = False
    actuator_context = None

    def audited_set_wheel_command(car, left_index, right_index, linear, angular):
        nonlocal actuator_context, sequence_id, frozen_after_process_stop
        actuator_context = (car, left_index, right_index)
        original(car, left_index, right_index, linear, angular)
        sequence_id += 1
        if frozen_after_process_stop:
            return
        reason = "fresh" if float(linear) != 0.0 or float(angular) != 0.0 else "zero"
        receipt_path = output / "fault_receipt.json"
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text())
            except (OSError, json.JSONDecodeError):
                receipt = {}
            if receipt.get("confirmed") and float(linear) == 0.0 and float(angular) == 0.0:
                if receipt.get("target") == "daemon":
                    reason = "watchdog_zero"
                    frozen_after_process_stop = True
                elif receipt.get("target") == "coordinator":
                    reason = "coordinator_fail_safe_zero"
                    frozen_after_process_stop = True
        ledger.update(sequence_id, time.monotonic_ns(), linear, angular, reason, os.getpid())

    def host_emergency_stop(_signal_number, _frame):
        nonlocal sequence_id, frozen_after_process_stop
        if actuator_context is None or frozen_after_process_stop:
            return
        car, left_index, right_index = actuator_context
        original(car, left_index, right_index, 0.0, 0.0)
        sequence_id += 1
        ledger.update(sequence_id, time.monotonic_ns(), 0.0, 0.0, "watchdog_zero", os.getpid())
        frozen_after_process_stop = True

    plant.set_wheel_command = audited_set_wheel_command
    previous_handler = signal.signal(signal.SIGUSR1, host_emergency_stop)
    try:
        plant.main()
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
        if not frozen_after_process_stop:
            ledger.update(sequence_id + 1, time.monotonic_ns(), 0.0, 0.0, "terminal_zero", os.getpid())
        ledger.close()


if __name__ == "__main__":
    main()
