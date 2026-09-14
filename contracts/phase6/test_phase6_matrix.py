#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE6 = ROOT / "contracts/phase6"


def load_matrix_module(testcase):
    path = PHASE6 / "phase6_matrix.py"
    testcase.assertTrue(path.is_file(), "Phase 6 matrix implementation is missing")
    spec = importlib.util.spec_from_file_location("phase6_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase6MatrixTests(unittest.TestCase):
    def test_contract_builds_exact_three_seed_four_scenario_matrix(self):
        contract_path = PHASE6 / "phase6_contract.json"
        self.assertTrue(contract_path.is_file(), "Phase 6 contract is missing")
        contract = json.loads(contract_path.read_text())
        module = load_matrix_module(self)

        cells = module.build_matrix(contract)

        self.assertEqual(len(cells), 12)
        self.assertEqual(len({cell["case_id"] for cell in cells}), 12)
        self.assertEqual({cell["seed"] for cell in cells}, set(contract["matrix"]["seeds"]))
        self.assertEqual({cell["scenario"] for cell in cells}, set(contract["matrix"]["scenarios"]))
        self.assertEqual(
            {cell["profile"] for cell in cells},
            {profile["name"] for profile in contract["matrix"]["profiles"]},
        )

    def test_aggregate_requires_every_cell_and_all_safety_gates(self):
        contract_path = PHASE6 / "phase6_contract.json"
        self.assertTrue(contract_path.is_file(), "Phase 6 contract is missing")
        contract = json.loads(contract_path.read_text())
        module = load_matrix_module(self)
        cells = module.build_matrix(contract)
        results = [
            {
                **cell,
                "reached": True,
                "static_collision_count": 0,
                "dynamic_collision_count": 0,
                "supervisor_aborts": 0,
                "solver_failures": 0,
                "candidate_valid_ratio_mean": 1.0,
                "sensor_to_wheel_p95_ms": 38.0,
                "path_error_p95_m": 0.05,
                "terminal_position_error_m": 0.05,
                "terminal_yaw_error_rad": 0.05,
                "wheel_command_delta_p95": 0.2,
            }
            for cell in cells
        ]
        inherited = {
            "hour_frames": 72000,
            "hour_duration_s": 3611.0,
            "hour_collision_count": 0,
            "maximum_fault_recovery_frames": 1,
        }

        passed = module.aggregate_matrix(results, inherited, contract["acceptance"])
        missing = module.aggregate_matrix(results[:-1], inherited, contract["acceptance"])
        collided_results = [dict(item) for item in results]
        collided_results[0]["static_collision_count"] = 1
        collided = module.aggregate_matrix(collided_results, inherited, contract["acceptance"])

        self.assertTrue(passed["gate_passed"])
        self.assertEqual(passed["reached_cases"], 12)
        self.assertFalse(missing["gate_passed"])
        self.assertFalse(collided["gate_passed"])

    def test_matrix_is_deterministic_and_perturbs_every_required_dimension(self):
        contract_path = PHASE6 / "phase6_contract.json"
        self.assertTrue(contract_path.is_file(), "Phase 6 contract is missing")
        contract = json.loads(contract_path.read_text())
        module = load_matrix_module(self)

        first = module.build_matrix(contract)
        second = module.build_matrix(contract)

        self.assertEqual(first, second)
        required = {
            "start_goal",
            "lighting",
            "material",
            "jpeg_rgb",
            "metric_depth",
            "camera_extrinsics",
            "dynamic_obstacle",
        }
        self.assertEqual(set(contract["matrix"]["perturbation_dimensions"]), required)
        for cell in first:
            self.assertEqual(set(cell["perturbations"]), required)


if __name__ == "__main__":
    unittest.main()
