#!/usr/bin/env python3
import unittest

from validate_phase5i import aggregate_runs


def summary(mode, latency, jitter, estop=0, watchdog=0, resets=0):
    return {
        "run_mode": mode,
        "frames": 180,
        "gate_passed": True,
        "metrics": {
            "sensor_to_wheel_ms": {"p95": latency},
            "sensor_to_wheel_jitter_ms": {"p95": jitter},
            "emergency_stop_frames": estop,
            "watchdog_stop_frames": watchdog,
            "reset_events": resets,
            "static_collision_count": 0,
            "dynamic_collision_count": 0,
        },
    }


class Phase5IValidationTests(unittest.TestCase):
    def test_aggregate_requires_all_three_distinct_passed_modes(self):
        runs = [
            summary("nominal", 29.0, 9.0),
            summary("emergency_stop_reset", 35.0, 12.0, estop=10, resets=1),
            summary("watchdog_reset", 32.0, 9.0, watchdog=4, resets=1),
        ]

        result = aggregate_runs(runs)

        self.assertEqual(result["run_modes"], ["nominal", "emergency_stop_reset", "watchdog_reset"])
        self.assertEqual(result["frames"], 540)
        self.assertEqual(result["maximum_sensor_to_wheel_p95_ms"], 35.0)
        self.assertEqual(result["maximum_jitter_p95_ms"], 12.0)
        self.assertEqual(result["emergency_stop_frames"], 10)
        self.assertEqual(result["watchdog_stop_frames"], 4)
        self.assertEqual(result["reset_events"], 2)
        self.assertTrue(result["all_run_gates_passed"])


if __name__ == "__main__":
    unittest.main()
