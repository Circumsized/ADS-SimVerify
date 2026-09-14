#!/usr/bin/env python3
import unittest

from validate_phase5k import aggregate_runs


class Phase5KValidationTests(unittest.TestCase):
    def test_aggregate_requires_all_five_modes_and_safety_invariants(self):
        runs = [
            {"run_mode": "hour_endurance", "frames": 72000, "gate_passed": True, "metrics": {"collision_count": 0}},
            {"run_mode": "coordinator_sigkill", "frames": 300, "gate_passed": True, "metrics": {"collision_count": 0}},
            {"run_mode": "daemon_sigkill", "frames": 81, "gate_passed": True, "metrics": {"collision_count": 0}},
            {"run_mode": "gpu_oom_recovery", "frames": 300, "gate_passed": True, "metrics": {"collision_count": 0}},
            {"run_mode": "disk_full_recovery", "frames": 300, "gate_passed": True, "metrics": {"collision_count": 0}},
        ]

        aggregate = aggregate_runs(runs)

        self.assertEqual(aggregate["run_modes"], [item["run_mode"] for item in runs])
        self.assertEqual(aggregate["frames"], 72981)
        self.assertEqual(aggregate["collision_count"], 0)
        self.assertTrue(aggregate["all_run_gates_passed"])


if __name__ == "__main__":
    unittest.main()
