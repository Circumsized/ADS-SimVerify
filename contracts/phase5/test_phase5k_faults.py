#!/usr/bin/env python3
import errno
import tempfile
import unittest
from pathlib import Path

from phase5k_faults import (
    SafetyLedger,
    discover_process_pid,
    episode_reset_due,
    fault_action,
    phase5j_run_mode,
    host_watchdog_due,
    watchdog_reset_due,
    write_enospc_probe,
)


class Phase5KFaultTests(unittest.TestCase):
    def test_fault_schedule_is_deterministic(self):
        self.assertEqual(fault_action("coordinator_sigkill", 80).kill_target, "coordinator")
        self.assertEqual(fault_action("daemon_sigkill", 80).kill_target, "daemon")
        self.assertTrue(fault_action("gpu_oom_recovery", 80).gpu_oom)
        self.assertTrue(fault_action("gpu_oom_recovery", 92).reset)
        self.assertTrue(fault_action("disk_full_recovery", 80).disk_full)
        self.assertTrue(fault_action("disk_full_recovery", 92).reset)
        self.assertFalse(fault_action("hour_endurance", 80).has_action)

    def test_process_discovery_uses_exact_dora_role_not_substring(self):
        table = [
            (10, "dora coordinator"),
            (11, "dora daemon"),
            (12, "python phase5k_fault_injector.py coordinator"),
        ]

        self.assertEqual(discover_process_pid("coordinator", table), 10)
        self.assertEqual(discover_process_pid("daemon", table), 11)

    def test_reserved_safety_ledger_keeps_latest_actual_wheel_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "safety.ledger"
            with SafetyLedger(path, size=4096) as ledger:
                ledger.update(7, 123456, 0.3, -0.2, "fresh", pid=42)
                ledger.update(8, 123999, 0.0, 0.0, "watchdog_zero", pid=42)
            state = SafetyLedger.read(path)

        self.assertEqual(state["sequence_id"], 8)
        self.assertEqual((state["linear"], state["angular"]), (0.0, 0.0))
        self.assertEqual(state["reason"], "watchdog_zero")
        self.assertEqual(state["pid"], 42)

    def test_dev_full_produces_real_enospc(self):
        result = write_enospc_probe(Path("/dev/full"))

        self.assertFalse(result["written"])
        self.assertEqual(result["errno"], errno.ENOSPC)

    def test_hour_mode_reuses_phase5j_four_scenario_schedule(self):
        self.assertEqual(phase5j_run_mode("hour_endurance"), "endurance")
        self.assertEqual(phase5j_run_mode("gpu_oom_recovery"), "gpu_oom_recovery")

    def test_host_watchdog_only_triggers_for_confirmed_daemon_loss_with_nonzero_actuator(self):
        receipt = {"target": "daemon", "confirmed": True}
        ledger = {"linear": 0.5, "angular": 0.1, "reason": "fresh"}

        self.assertTrue(host_watchdog_due(receipt, ledger))
        self.assertFalse(host_watchdog_due({**receipt, "confirmed": False}, ledger))
        self.assertFalse(host_watchdog_due({**receipt, "target": "coordinator"}, ledger))
        self.assertFalse(host_watchdog_due(receipt, {**ledger, "linear": 0.0, "angular": 0.0}))

    def test_episode_reset_only_occurs_when_a_known_episode_changes(self):
        self.assertFalse(episode_reset_due(None, 0))
        self.assertFalse(episode_reset_due(3, 3))
        self.assertTrue(episode_reset_due(3, 4))

    def test_watchdog_reset_is_limited_to_missing_input_fault_with_cooldown(self):
        self.assertTrue(watchdog_reset_due("hour_endurance", "fault", "missing_runtime_input", 100, None))
        self.assertFalse(watchdog_reset_due("hour_endurance", "fault", "runtime_unhealthy", 100, None))
        self.assertFalse(watchdog_reset_due("hour_endurance", "active", "allow", 100, None))
        self.assertFalse(watchdog_reset_due("hour_endurance", "fault", "missing_runtime_input", 102, 100))
        self.assertTrue(watchdog_reset_due("hour_endurance", "fault", "missing_runtime_input", 104, 100))


if __name__ == "__main__":
    unittest.main()
