#!/usr/bin/env python3
import unittest

from phase5i_evidence_sink import gate_status, localization_sequence_valid, run_metrics


def row(frame_id, state, reason="allow", latency=40.0, reset=False):
    return {
        "source_frame_id": frame_id,
        "safety_state": state,
        "safety_reason": reason,
        "reset_applied": int(reset),
        "command_linear_mps": 0.0 if state != "active" else 0.3,
        "command_angular_radps": 0.0,
        "sensor_to_wheel_ms": latency,
        "static_collision": 0,
        "dynamic_collision": 0,
        "actuator_reason": "fresh",
    }


class Phase5IEvidenceTests(unittest.TestCase):
    def test_localization_evidence_is_native_two_hz_sequence(self):
        expected = list(range(0, 180, 10))
        self.assertTrue(localization_sequence_valid(expected, [True] * 18, 180, 20, 2))
        self.assertFalse(localization_sequence_valid(expected[:-1], [True] * 17, 180, 20, 2))
        self.assertFalse(localization_sequence_valid(expected, [True] * 17 + [False], 180, 20, 2))

    def test_smoke_result_never_claims_or_rejects_the_formal_gate(self):
        self.assertEqual(gate_status(False, True), "smoke_only")
        self.assertEqual(gate_status(False, False), "smoke_only")
        self.assertEqual(gate_status(True, True), "run_gate_passed")
        self.assertEqual(gate_status(True, False), "run_gate_rejected")

    def test_metrics_count_startup_estop_reset_recovery_and_jitter(self):
        rows = [
            row(0, "warmup", "warming_up", 35.0),
            row(1, "ready", "startup_ready", 37.0),
            row(2, "active", latency=39.0),
            row(3, "emergency_stop", "emergency_stop", 41.0),
            row(4, "warmup", "warming_up", 38.0, reset=True),
            row(5, "ready", "startup_ready", 40.0),
            row(6, "active", latency=42.0),
        ]

        metrics = run_metrics(rows)

        self.assertTrue(metrics["exact_source_frame_ids"])
        self.assertEqual(metrics["startup_zero_frames"], 2)
        self.assertEqual(metrics["emergency_stop_frames"], 1)
        self.assertEqual(metrics["reset_events"], 1)
        self.assertTrue(metrics["active_after_last_reset"])
        self.assertAlmostEqual(metrics["sensor_to_wheel_ms"]["p95"], 41.7)
        self.assertAlmostEqual(metrics["sensor_to_wheel_jitter_ms"]["p95"], 2.75)

    def test_metrics_recognize_supervisor_and_actuator_watchdogs(self):
        rows = [row(0, "active"), row(1, "fault", "stale_command")]
        rows.append({**row(2, "fault"), "actuator_reason": "actuator_watchdog_timeout"})

        metrics = run_metrics(rows)

        self.assertEqual(metrics["supervisor_watchdog_frames"], 1)
        self.assertEqual(metrics["actuator_watchdog_frames"], 1)
        self.assertEqual(metrics["watchdog_stop_frames"], 2)


if __name__ == "__main__":
    unittest.main()
