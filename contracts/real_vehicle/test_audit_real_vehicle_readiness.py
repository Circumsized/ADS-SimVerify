#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDITOR = ROOT / "contracts/real_vehicle/audit_real_vehicle_readiness.py"
CONTRACT = ROOT / "contracts/real_vehicle/real_vehicle_acceptance_contract.json"


def load_auditor(testcase):
    testcase.assertTrue(AUDITOR.is_file(), "real-vehicle readiness auditor is missing")
    spec = importlib.util.spec_from_file_location("audit_real_vehicle_readiness", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RealVehicleReadinessAuditTests(unittest.TestCase):
    def test_contract_and_implementation_thresholds_do_not_drift(self):
        module = load_auditor(self)
        self.assertTrue(CONTRACT.is_file(), "real-vehicle acceptance contract is missing")
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

        self.assertEqual(contract["thresholds"], module.DEFAULT_THRESHOLDS)

    def test_absent_camera_device_overrides_claimed_camera_pass(self):
        module = load_auditor(self)
        supplied = {
            "camera": {
                "device_present": True,
                "calibration_gate_passed": True,
                "paired_frame_ratio": 1.0,
                "depth_scale_relative_error": 0.0,
            }
        }

        evidence = module.build_preflight_evidence("nodes: []", [], supplied)

        self.assertFalse(evidence["camera"]["device_present"])
        self.assertFalse(evidence["camera"]["calibration_gate_passed"])
        self.assertEqual(evidence["camera"]["detected_devices"], [])


if __name__ == "__main__":
    unittest.main()
