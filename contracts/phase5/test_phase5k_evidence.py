#!/usr/bin/env python3
import unittest

from phase5k_evidence import (
    coordinator_gate,
    daemon_gate,
    hour_gate,
    out_of_band_stop_gate,
    recovery_gate,
    resource_receipt_gate,
)


class Phase5KEvidenceTests(unittest.TestCase):
    def test_hour_gate_requires_exact_hour_scenarios_latency_and_zero_collisions(self):
        metrics = {
            "frames": 72000,
            "wall_duration_s": 3600.0,
            "exact_source_frame_ids": True,
            "scenario_coverage": ["straight_aisle", "diagonal_turn", "pallet_detour", "crossing_cart"],
            "collision_count": 0,
            "wrong_generation_commands": 0,
            "active_ratio": 0.99,
            "sensor_to_wheel_p95_ms": 45.0,
        }
        self.assertTrue(hour_gate(metrics, 72000, 3600.0, 0.98, 50.0))
        self.assertFalse(hour_gate({**metrics, "frames": 71999}, 72000, 3600.0, 0.98, 50.0))
        self.assertFalse(hour_gate({**metrics, "wall_duration_s": 3599.9}, 72000, 3600.0, 0.98, 50.0))
        self.assertFalse(hour_gate({**metrics, "active_ratio": 0.979}, 72000, 3600.0, 0.98, 50.0))
        self.assertFalse(hour_gate({**metrics, "sensor_to_wheel_p95_ms": 50.1}, 72000, 3600.0, 0.98, 50.0))

    def test_out_of_band_gate_requires_confirmed_kill_and_zero_ledger(self):
        receipt = {"confirmed": True, "kill_monotonic_ns": 1_000_000_000}
        ledger = {
            "linear": 0.0,
            "angular": 0.0,
            "monotonic_ns": 1_090_000_000,
            "reason": "watchdog_zero",
        }
        self.assertTrue(out_of_band_stop_gate(receipt, ledger, 100.0))
        self.assertFalse(out_of_band_stop_gate(receipt, {**ledger, "linear": 0.1}, 100.0))
        self.assertFalse(out_of_band_stop_gate(receipt, {**ledger, "monotonic_ns": 1_101_000_000}, 100.0))

    def test_recovery_gate_requires_real_fault_stop_reset_and_active_recovery(self):
        metrics = {
            "fault_observed": True,
            "maximum_fault_stop_latency_frames": 2,
            "reset_events": 1,
            "active_after_last_reset": True,
            "collision_count": 0,
        }
        self.assertTrue(recovery_gate(metrics, 3))
        self.assertFalse(recovery_gate({**metrics, "fault_observed": False}, 3))
        self.assertFalse(recovery_gate({**metrics, "maximum_fault_stop_latency_frames": 4}, 3))

    def test_resource_receipts_require_real_cuda_oom_or_real_enospc(self):
        gpu = {
            "observed": True,
            "message_contains_cuda_oom": True,
            "allocated_bytes_before_failure": 1024,
            "free_bytes_before": 4096,
            "free_bytes_after_release": 4000,
        }
        disk = {"written": False, "errno": 28}

        self.assertTrue(resource_receipt_gate("gpu_oom_recovery", gpu))
        self.assertFalse(resource_receipt_gate("gpu_oom_recovery", {**gpu, "observed": False}))
        self.assertTrue(resource_receipt_gate("disk_full_recovery", disk))
        self.assertFalse(resource_receipt_gate("disk_full_recovery", {**disk, "errno": 5}))

    def test_coordinator_gate_requires_confirmed_kill_and_local_fail_safe_stop(self):
        metrics = {
            "frames": 172,
            "exact_source_frame_ids": True,
            "collision_count": 0,
            "wrong_generation_commands": 0,
        }
        receipt = {
            "target": "coordinator",
            "confirmed": True,
            "kill_monotonic_ns": 1_000_000_000,
        }
        ledger = {
            "linear": 0.0,
            "angular": 0.0,
            "monotonic_ns": 1_080_000_000,
            "reason": "coordinator_fail_safe_zero",
        }

        self.assertTrue(coordinator_gate(metrics, receipt, ledger, 100.0))
        self.assertFalse(coordinator_gate(metrics, {**receipt, "confirmed": False}, ledger, 100.0))
        self.assertFalse(coordinator_gate(metrics, receipt, {**ledger, "linear": 0.2}, 100.0))

    def test_daemon_gate_uses_out_of_band_actual_actuator_ledger(self):
        metrics = {
            "frames": 81,
            "exact_source_frame_ids": True,
            "collision_count": 0,
            "wrong_generation_commands": 0,
        }
        receipt = {
            "target": "daemon",
            "confirmed": True,
            "kill_monotonic_ns": 1_000_000_000,
        }
        ledger = {
            "linear": 0.0,
            "angular": 0.0,
            "monotonic_ns": 1_090_000_000,
            "reason": "watchdog_zero",
        }

        self.assertTrue(daemon_gate(metrics, receipt, ledger, 100.0))
        self.assertFalse(daemon_gate(metrics, receipt, {**ledger, "linear": 0.2}, 100.0))


if __name__ == "__main__":
    unittest.main()
