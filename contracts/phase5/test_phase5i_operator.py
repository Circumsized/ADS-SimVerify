#!/usr/bin/env python3
import unittest

from phase5i_operator_safety_node import safety_request_for


class Phase5IOperatorTests(unittest.TestCase):
    def test_nominal_run_never_requests_stop_or_reset(self):
        self.assertEqual({safety_request_for("nominal", tick) for tick in range(100)}, {(False, False)})

    def test_emergency_stop_trial_has_a_stop_window_and_single_reset(self):
        self.assertEqual(safety_request_for("emergency_stop_reset", 39), (False, False))
        self.assertEqual(safety_request_for("emergency_stop_reset", 40), (True, False))
        self.assertEqual(safety_request_for("emergency_stop_reset", 49), (True, False))
        self.assertEqual(safety_request_for("emergency_stop_reset", 50), (False, True))
        self.assertEqual(safety_request_for("emergency_stop_reset", 51), (False, False))

    def test_watchdog_trial_resets_after_the_injected_stale_frame(self):
        self.assertEqual(safety_request_for("watchdog_reset", 64), (False, False))
        self.assertEqual(safety_request_for("watchdog_reset", 65), (False, True))
        self.assertEqual(safety_request_for("watchdog_reset", 66), (False, False))


if __name__ == "__main__":
    unittest.main()
