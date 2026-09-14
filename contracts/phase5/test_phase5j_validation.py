#!/usr/bin/env python3
import unittest
import tempfile
from pathlib import Path
import hashlib

from validate_phase5j import aggregate_runs, telemetry_reference_valid


def summary(mode, generation=1):
    return {
        "run_mode": mode,
        "gate_passed": True,
        "frames": 300,
        "controller_generation": generation,
        "controller_pid": 100 + generation,
        "metrics": {
            "collision_count": 0,
            "maximum_fault_stop_latency_frames": 3 if "sigkill" in mode else 0,
            "wrong_generation_commands": 0,
            "active_after_last_reset": mode in {"sensor_freeze", "restart_recovery"},
            "scenario_coverage": ["straight_aisle"],
        },
    }


class Phase5JValidationTests(unittest.TestCase):
    def test_aggregate_requires_all_fault_modes_and_fresh_restart_pid(self):
        endurance = summary("endurance")
        endurance["frames"] = 4400
        endurance["metrics"]["scenario_coverage"] = [
            "straight_aisle", "diagonal_turn", "pallet_detour", "crossing_cart"
        ]
        runs = [
            endurance,
            summary("controller_sigkill"),
            summary("supervisor_sigkill"),
            summary("sensor_freeze"),
            summary("restart_recovery", generation=2),
        ]

        aggregate = aggregate_runs(runs)

        self.assertTrue(aggregate["all_run_gates_passed"])
        self.assertEqual(aggregate["run_modes"], [item["run_mode"] for item in runs])
        self.assertTrue(aggregate["restart_used_new_controller_process"])

    def test_telemetry_reference_requires_matching_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            telemetry = Path(directory) / "frames.csv"
            telemetry.write_bytes(b"frame\n1\n")
            summary = {
                "telemetry": telemetry.name,
                "telemetry_sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
            }
            self.assertTrue(telemetry_reference_valid(summary_path, summary))
            telemetry.write_bytes(b"changed")
            self.assertFalse(telemetry_reference_valid(summary_path, summary))


if __name__ == "__main__":
    unittest.main()
