#!/usr/bin/env python3
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "contracts/phase6/run_phase6_matrix.py"


def load_runner(testcase):
    testcase.assertTrue(RUNNER.is_file(), "Phase 6 Isaac matrix runner is missing")
    spec = importlib.util.spec_from_file_location("run_phase6_matrix", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase6RunnerTests(unittest.TestCase):
    def test_finalize_matrix_writes_summary_before_runtime_shutdown(self):
        module = load_runner(self)
        self.assertTrue(
            hasattr(module, "finalize_matrix"),
            "Phase 6 must finalize evidence before Isaac runtime shutdown",
        )
        contract = json.loads((ROOT / "contracts/phase6/phase6_contract.json").read_text())
        cells = module.build_matrix(contract)
        results = [{
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
        } for cell in cells]
        inherited = {
            "hour_frames": 72000, "hour_duration_s": 3611.0,
            "hour_collision_count": 0, "maximum_fault_recovery_frames": 1,
        }
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in cells]
        with tempfile.TemporaryDirectory() as temp:
            result = module.finalize_matrix(
                Path(temp), images, results, inherited, contract, formal=True
            )

            written = json.loads((Path(temp) / "summary.json").read_text())

        self.assertTrue(result["gate_passed"])
        self.assertEqual(written["aggregate"], result["aggregate"])

    def test_rgb_and_depth_perturbations_are_seed_deterministic(self):
        module = load_runner(self)
        image = np.full((24, 32, 3), 120, dtype=np.uint8)
        depth = np.full((24, 32), 2.0, dtype=np.float32)
        rgb_config = {
            "rgb_gain": [0.8, 1.2], "rgb_offset": [-8.0, 8.0], "gamma": [0.9, 1.1],
            "gaussian_noise_sigma_u8": [2.0, 2.0], "motion_blur_probability": 0.5,
            "jpeg_quality": [80, 80],
        }
        depth_config = {
            "scale": [0.99, 1.01], "noise_sigma_m": [0.003, 0.003],
            "dropout_probability": [0.01, 0.01],
        }

        rgb_a = module.perturb_rgb(image, np.random.default_rng(7), rgb_config)
        rgb_b = module.perturb_rgb(image, np.random.default_rng(7), rgb_config)
        depth_a = module.perturb_depth(depth, np.random.default_rng(9), depth_config)
        depth_b = module.perturb_depth(depth, np.random.default_rng(9), depth_config)

        np.testing.assert_array_equal(rgb_a, rgb_b)
        np.testing.assert_array_equal(depth_a, depth_b)
        self.assertEqual(rgb_a.shape, image.shape)
        self.assertFalse(np.array_equal(rgb_a, image))
        self.assertTrue(np.array_equal(np.isfinite(depth_a), np.isfinite(depth_b)))

    def test_scenario_offsets_preserve_name_and_velocity(self):
        module = load_runner(self)
        base = module.Scenario("route", (1.0, 2.0, 0.1, 0.2), (3.0, 4.0, 0.3))
        cell = {"perturbations": {"start_goal": {
            "start_dx_m": 0.05, "start_dy_m": -0.04, "start_dyaw_rad": 0.02,
            "goal_dx_m": -0.03, "goal_dy_m": 0.06, "goal_dyaw_rad": -0.01,
        }}}

        result = module.perturbed_scenario(base, cell)

        self.assertEqual(result.name, "route")
        self.assertEqual(result.start, (1.05, 1.96, 0.12000000000000001, 0.2))
        self.assertEqual(result.goal, (2.97, 4.06, 0.29))

    def test_wheel_command_delta_p95_reads_angular_command_changes(self):
        module = load_runner(self)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "telemetry.csv"
            with path.open("w", newline="", encoding="ascii") as target:
                writer = csv.DictWriter(target, fieldnames=["omega_radps"])
                writer.writeheader()
                for value in [0.0, 0.1, 0.3, 0.2, 0.2]:
                    writer.writerow({"omega_radps": value})

            result = module.wheel_command_delta_p95(path)

        self.assertAlmostEqual(result, 0.185)


if __name__ == "__main__":
    unittest.main()
