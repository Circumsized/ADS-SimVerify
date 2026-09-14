#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "contracts/real_vehicle/validate_pre_hardware.py"


def load_validator(testcase):
    testcase.assertTrue(VALIDATOR.is_file(), "pre-hardware validator is missing")
    spec = importlib.util.spec_from_file_location("validate_pre_hardware", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreHardwareValidationTests(unittest.TestCase):
    def test_blocked_summary_is_valid_only_when_all_six_gates_are_explicit(self):
        module = load_validator(self)
        summary = {
            "status": "real_vehicle_gate_blocked",
            "blocked_gates": list(module.REQUIRED_GATES),
            "gate_passed": False,
            "real_vehicle_control_allowed": False,
        }

        self.assertEqual(module.summary_errors(summary), [])

    def test_validator_rejects_false_promotion_or_missing_gate(self):
        module = load_validator(self)
        summary = {
            "status": "real_vehicle_gate_passed",
            "blocked_gates": ["camera"],
            "gate_passed": True,
            "real_vehicle_control_allowed": True,
        }

        errors = module.summary_errors(summary)

        self.assertTrue(errors)
        self.assertTrue(any("control" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
