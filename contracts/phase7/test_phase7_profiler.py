#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROFILER = ROOT / "contracts/phase7/run_phase7_profile.py"


def load_profiler(testcase):
    testcase.assertTrue(PROFILER.is_file(), "Phase 7 runtime profiler is missing")
    spec = importlib.util.spec_from_file_location("run_phase7_profile", PROFILER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase7ProfilerTests(unittest.TestCase):
    def test_xfeat_preprocessing_letterboxes_and_normalizes_native_frame(self):
        module = load_profiler(self)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:, :, 1] = np.arange(640, dtype=np.uint8)[None, :]

        tensor = module.prepare_xfeat_tensor(image)

        self.assertEqual(tensor.shape, (1, 1, 640, 640))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(float(tensor.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(tensor.std()), 1.0, places=5)
        self.assertTrue(np.all(np.isfinite(tensor)))

    def test_nvidia_sampler_parser_reports_peak_and_baseline_adjusted_vram(self):
        module = load_profiler(self)
        lines = [
            "2026/07/16 10:00:00.000, 900, 10, 15.0, 45",
            "2026/07/16 10:00:00.100, 1500, 80, 75.0, 62",
            "2026/07/16 10:00:00.200, 1300, 50, 55.0, 58",
        ]

        result = module.summarize_nvidia_samples(lines, baseline_vram_mib=800.0)

        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["vram_peak_mib"], 1500.0)
        self.assertEqual(result["vram_incremental_peak_mib"], 700.0)
        self.assertEqual(result["gpu_utilization_peak_percent"], 80.0)
        self.assertEqual(result["gpu_power_peak_w"], 75.0)
        self.assertEqual(result["gpu_temperature_peak_c"], 62.0)

    def test_semantic_preprocessing_matches_model_contract(self):
        module = load_profiler(self)
        image = np.full((480, 640, 3), 127, dtype=np.uint8)

        tensor = module.prepare_semantic_tensor(image)

        self.assertEqual(tensor.shape, (1, 3, 240, 320))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(tensor)))

    def test_inherited_runtime_evidence_reads_frozen_status_shapes(self):
        module = load_profiler(self)

        evidence = module.inherited_runtime_evidence()

        self.assertEqual(evidence["phase5k_hour_frames"], 72000)
        self.assertEqual(evidence["phase5k_maximum_fault_recovery_frames"], 1)
        self.assertGreater(evidence["phase5d_dora_perception_p95_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
