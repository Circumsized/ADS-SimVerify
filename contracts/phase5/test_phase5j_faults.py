#!/usr/bin/env python3
import unittest

from phase5j_faults import CommandGenerationGuard, SensorSequenceGuard, fault_action


class Phase5JFaultTests(unittest.TestCase):
    def test_sensor_guard_rejects_frozen_replayed_and_old_generation_frames(self):
        guard = SensorSequenceGuard(expected_generation=2)

        self.assertEqual(guard.observe(2, 0, 1000.0), (True, "fresh"))
        self.assertEqual(guard.observe(2, 1, 1050.0), (True, "fresh"))
        self.assertEqual(guard.observe(2, 1, 1050.0), (False, "sensor_replay"))
        self.assertEqual(guard.observe(1, 2, 1100.0), (False, "wrong_generation"))
        self.assertEqual(guard.observe(2, 2, 1040.0), (False, "stale_sensor_timestamp"))

    def test_command_guard_rejects_replay_and_previous_process_generation(self):
        guard = CommandGenerationGuard(expected_generation=3)

        self.assertTrue(guard.accept(3, 0))
        self.assertTrue(guard.accept(3, 1))
        self.assertFalse(guard.accept(3, 1))
        self.assertFalse(guard.accept(2, 2))

    def test_fault_schedule_is_exact_and_deterministic(self):
        self.assertEqual(fault_action("controller_sigkill", 80).kill_target, "controller")
        self.assertEqual(fault_action("supervisor_sigkill", 80).kill_target, "supervisor")
        self.assertTrue(fault_action("sensor_freeze", 80).freeze_sensor)
        self.assertFalse(fault_action("sensor_freeze", 88).freeze_sensor)
        self.assertTrue(fault_action("sensor_freeze", 92).reset)
        self.assertFalse(fault_action("endurance", 80).has_action)


if __name__ == "__main__":
    unittest.main()
