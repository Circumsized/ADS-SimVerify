#!/usr/bin/env python3
import unittest

from phase5j_evidence_sink import common_gate_passes, fault_stop_latency, run_metrics


def row(frame_id, scenario="straight_aisle", linear=0.3, reason="fresh", generation=1):
    return {
        "source_frame_id": frame_id,
        "scenario_name": scenario,
        "command_linear_mps": linear,
        "command_angular_radps": 0.0,
        "actuator_reason": reason,
        "controller_generation": generation,
        "sensor_frozen": 0,
        "reset_applied": 0,
        "safety_state": "active" if linear else "fault",
        "static_collision": 0,
        "dynamic_collision": 0,
        "sensor_to_wheel_ms": 30.0,
    }


class Phase5JEvidenceTests(unittest.TestCase):
    def test_fault_stop_latency_counts_until_first_zero_wheel_command(self):
        rows = [row(index, linear=0.3 if index < 13 else 0.0) for index in range(20)]

        self.assertEqual(fault_stop_latency(rows, event_frame=10), 3)

    def test_metrics_cover_all_scenarios_and_reject_collisions_or_old_generation(self):
        names = ("straight_aisle", "diagonal_turn", "pallet_detour", "crossing_cart")
        rows = [row(index, names[index % 4], generation=2) for index in range(16)]
        metrics = run_metrics(rows, expected_generation=2)

        self.assertEqual(metrics["scenario_coverage"], list(names))
        self.assertTrue(metrics["exact_source_frame_ids"])
        self.assertEqual(metrics["wrong_generation_commands"], 0)
        self.assertEqual(metrics["collision_count"], 0)

        rows[8]["controller_generation"] = 1
        rows[9]["static_collision"] = 1
        metrics = run_metrics(rows, expected_generation=2)
        self.assertEqual(metrics["wrong_generation_commands"], 1)
        self.assertEqual(metrics["collision_count"], 1)

    def test_metrics_count_frozen_sensor_window_and_post_reset_recovery(self):
        rows = [row(index, linear=0.0 if 4 <= index <= 8 else 0.3) for index in range(12)]
        for index in range(4, 8):
            rows[index]["sensor_frozen"] = 1
        rows[8]["reset_applied"] = 1

        metrics = run_metrics(rows, expected_generation=1)

        self.assertEqual(metrics["sensor_frozen_frames"], 4)
        self.assertEqual(metrics["reset_events"], 1)
        self.assertTrue(metrics["active_after_last_reset"])

    def test_actuator_watchdog_zero_is_not_counted_as_active_motion(self):
        rows = [row(0, linear=0.0, reason="actuator_watchdog_timeout")]
        rows[0]["safety_state"] = "active"

        metrics = run_metrics(rows, expected_generation=1)

        self.assertEqual(metrics["active_frames"], 0)

    def test_common_gate_rejects_latency_collision_generation_and_episode_failures(self):
        metrics = {
            "exact_source_frame_ids": True,
            "scenario_coverage": ["straight_aisle"],
            "episodes": 1,
            "wrong_generation_commands": 0,
            "collision_count": 0,
            "sensor_to_wheel_p95_ms": 49.0,
        }
        self.assertTrue(common_gate_passes(metrics, ["straight_aisle"], 1, 50.0))
        self.assertFalse(common_gate_passes({**metrics, "sensor_to_wheel_p95_ms": 50.1}, ["straight_aisle"], 1, 50.0))
        self.assertFalse(common_gate_passes({**metrics, "episodes": 0}, ["straight_aisle"], 1, 50.0))


if __name__ == "__main__":
    unittest.main()
