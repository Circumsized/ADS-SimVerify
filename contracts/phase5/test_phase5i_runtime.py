#!/usr/bin/env python3
import unittest

from phase5i_runtime import (
    ActuatorWatchdog,
    dual_resolution_geometry,
    localization_due,
    proposal_timestamp_ms,
)


class Phase5IRuntimeTests(unittest.TestCase):
    def test_dual_resolution_keeps_sensor_contract_and_scales_control_intrinsics(self):
        source = {
            "image_size": [640, 480],
            "intrinsics": {"fx": 204.0, "fy": 154.0, "cx": 319.5, "cy": 239.5},
        }

        sensor, control = dual_resolution_geometry(source)

        self.assertEqual(sensor["image_size"], [640, 480])
        self.assertEqual(sensor["intrinsics"], source["intrinsics"])
        self.assertEqual(control["image_size"], [320, 240])
        self.assertAlmostEqual(control["intrinsics"]["fx"] / 320, source["intrinsics"]["fx"] / 640)
        self.assertAlmostEqual(control["intrinsics"]["fy"] / 240, source["intrinsics"]["fy"] / 480)
        self.assertEqual(source["image_size"], [640, 480])

    def test_actuator_watchdog_zeros_stale_safe_command(self):
        watchdog = ActuatorWatchdog(timeout_ms=150)
        self.assertTrue(watchdog.update(1, 0.0, 0.3, 0.2))

        fresh = watchdog.command(150.0)
        stale = watchdog.command(150.001)

        self.assertEqual((fresh.linear, fresh.angular, fresh.reason), (0.3, 0.2, "fresh"))
        self.assertEqual((stale.linear, stale.angular, stale.reason), (0.0, 0.0, "actuator_watchdog_timeout"))

    def test_actuator_watchdog_rejects_replayed_or_invalid_safe_command(self):
        watchdog = ActuatorWatchdog(timeout_ms=150)
        self.assertTrue(watchdog.update(4, 100.0, 0.4, -0.3))
        self.assertFalse(watchdog.update(4, 110.0, 0.2, 0.0))
        self.assertFalse(watchdog.update(3, 120.0, 0.2, 0.0))
        self.assertFalse(watchdog.update(5, 120.0, -0.1, 0.0))
        self.assertFalse(watchdog.update(5, 120.0, 0.2, 0.7))

        command = watchdog.command(200.0)
        self.assertEqual((command.linear, command.angular), (0.4, -0.3))

    def test_watchdog_trial_injects_one_reproducibly_stale_proposal(self):
        self.assertEqual(proposal_timestamp_ms("nominal", 60, 5000.0, 150.0), 5000.0)
        self.assertEqual(proposal_timestamp_ms("watchdog_reset", 59, 5000.0, 150.0), 5000.0)
        self.assertLess(proposal_timestamp_ms("watchdog_reset", 60, 5000.0, 150.0), 4850.0)
        self.assertEqual(proposal_timestamp_ms("watchdog_reset", 61, 5000.0, 150.0), 5000.0)

    def test_native_localization_capture_runs_at_two_hz_outside_twenty_hz_control(self):
        due = [frame for frame in range(40) if localization_due(frame, 20, 2)]
        self.assertEqual(due, [0, 10, 20, 30])


if __name__ == "__main__":
    unittest.main()
