#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "contracts/real_vehicle/real_vehicle_acceptance.py"


def load_module(testcase):
    testcase.assertTrue(MODULE.is_file(), "real-vehicle acceptance logic is missing")
    spec = importlib.util.spec_from_file_location("real_vehicle_acceptance", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passing_evidence():
    return {
        "topology": {
            "forbidden_runtime_dependencies": [],
            "global_map_source": "real_slam_occupancy_map",
            "independent_safety_source": "independent_metric_depth_guard",
        },
        "camera": {
            "device_present": True,
            "calibration_gate_passed": True,
            "paired_frame_ratio": 0.995,
            "depth_scale_relative_error": 0.01,
        },
        "localization": {
            "duration_s": 900.0,
            "ate_rmse_m": 0.06,
            "rpe_translation_p95_m": 0.04,
            "yaw_error_p95_deg": 2.0,
            "dropout_ratio": 0.005,
            "relocalization_success_ratio": 1.0,
            "relocalization_p95_s": 1.2,
        },
        "global_planning": {
            "map_hash_present": True,
            "routes": 8,
            "route_success_ratio": 1.0,
            "invalid_or_occupied_waypoints": 0,
        },
        "collision_supervisor": {
            "physical_cases": 40,
            "stop_recall": 1.0,
            "go_specificity": 0.98,
            "minimum_stop_margin_m": 0.15,
            "oracle_used": False,
        },
        "actuator": {
            "wheels_off_ground_tested": True,
            "command_sign_correct": True,
            "velocity_tracking_p95_mps": 0.04,
            "yaw_rate_tracking_p95_radps": 0.08,
            "watchdog_stop_p95_ms": 80.0,
            "emergency_stop_p95_ms": 45.0,
            "zero_command_creep_mps": 0.005,
        },
    }


class RealVehicleAcceptanceTests(unittest.TestCase):
    def test_simulation_topology_is_rejected_as_real_vehicle_topology(self):
        module = load_module(self)
        topology = (ROOT / "dora_dataflow.yaml").read_text(encoding="utf-8")

        result = module.audit_topology_text(topology)

        self.assertFalse(result["gate_passed"])
        self.assertIn("isaac_sim_env", result["forbidden_runtime_dependencies"])

    def test_missing_hardware_evidence_blocks_every_physical_gate(self):
        module = load_module(self)

        result = module.evaluate_acceptance({}, module.DEFAULT_THRESHOLDS)

        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["real_vehicle_control_allowed"])
        self.assertEqual(
            set(result["blocked_gates"]),
            {"topology", "camera", "localization", "global_planning", "collision_supervisor", "actuator"},
        )

    def test_all_independent_real_world_gates_must_pass_together(self):
        module = load_module(self)
        evidence = passing_evidence()

        passed = module.evaluate_acceptance(evidence, module.DEFAULT_THRESHOLDS)
        evidence["actuator"]["emergency_stop_p95_ms"] = 151.0
        failed = module.evaluate_acceptance(evidence, module.DEFAULT_THRESHOLDS)

        self.assertTrue(passed["gate_passed"])
        self.assertTrue(passed["real_vehicle_control_allowed"])
        self.assertFalse(failed["gate_passed"])
        self.assertIn("actuator", failed["blocked_gates"])

    def test_collision_guard_must_not_reuse_learned_perception_output(self):
        module = load_module(self)
        evidence = passing_evidence()
        evidence["topology"]["independent_safety_source"] = "metric_depth_perception_bev"

        result = module.evaluate_acceptance(evidence, module.DEFAULT_THRESHOLDS)

        self.assertFalse(result["gate_passed"])
        self.assertIn("topology", result["blocked_gates"])


if __name__ == "__main__":
    unittest.main()
