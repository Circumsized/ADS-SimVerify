"""Unit tests for the actuator-side control command gate.

These tests import only ``control_gate`` (no Isaac / Dora / numpy), so they
run in any plain Python environment and are safe to wire into CI.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import control_gate


class ParseControlCommandTests(unittest.TestCase):
    def test_accepts_in_range_command(self):
        self.assertEqual(control_gate.parse_control_command([0.5, 0.2]), (0.5, 0.2))

    def test_accepts_boundary_values(self):
        self.assertEqual(
            control_gate.parse_control_command(
                [control_gate.CONTROL_V_MAX, control_gate.CONTROL_W_MIN]
            ),
            (control_gate.CONTROL_V_MAX, control_gate.CONTROL_W_MIN),
        )

    def test_rejects_wrong_length(self):
        self.assertIsNone(control_gate.parse_control_command([0.5]))
        self.assertIsNone(control_gate.parse_control_command([0.5, 0.2, 0.1]))
        self.assertIsNone(control_gate.parse_control_command([]))

    def test_rejects_none(self):
        self.assertIsNone(control_gate.parse_control_command(None))

    def test_rejects_non_finite(self):
        self.assertIsNone(control_gate.parse_control_command([float("nan"), 0.2]))
        self.assertIsNone(control_gate.parse_control_command([0.5, float("inf")]))
        self.assertIsNone(control_gate.parse_control_command([0.5, float("-inf")]))

    def test_rejects_out_of_range(self):
        self.assertIsNone(control_gate.parse_control_command([1.5, 0.2]))
        self.assertIsNone(control_gate.parse_control_command([-0.1, 0.2]))
        self.assertIsNone(control_gate.parse_control_command([0.5, 5.0]))
        self.assertIsNone(control_gate.parse_control_command([0.5, -5.0]))

    def test_rejects_non_numeric(self):
        self.assertIsNone(control_gate.parse_control_command(["a", "b"]))


class IsControlFreshTests(unittest.TestCase):
    def test_none_is_never_fresh(self):
        self.assertFalse(control_gate.is_control_fresh(None, 1000.0))

    def test_within_window_is_fresh(self):
        self.assertTrue(control_gate.is_control_fresh(10.0, 10.0 + 0.150))

    def test_beyond_window_is_stale(self):
        self.assertFalse(control_gate.is_control_fresh(10.0, 10.0 + 0.250))

    def test_exact_boundary_is_fresh(self):
        boundary = control_gate.CONTROL_STALE_MS / 1000.0
        self.assertTrue(control_gate.is_control_fresh(10.0, 10.0 + boundary))


class ContractEnvelopeTests(unittest.TestCase):
    def test_envelope_matches_frozen_contract_union(self):
        self.assertEqual(control_gate.CONTROL_V_MIN, 0.0)
        self.assertEqual(control_gate.CONTROL_V_MAX, 0.8)
        self.assertEqual(control_gate.CONTROL_W_MIN, -1.0)
        self.assertEqual(control_gate.CONTROL_W_MAX, 1.0)
        self.assertEqual(control_gate.CONTROL_STALE_MS, 200.0)

    def test_no_forbidden_literals(self):
        self.assertTrue(math.isfinite(control_gate.CONTROL_V_MAX))


if __name__ == "__main__":
    unittest.main()
