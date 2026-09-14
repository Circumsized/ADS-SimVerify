#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE7 = ROOT / "contracts/phase7"


def load_profile(testcase):
    path = PHASE7 / "phase7_profile.py"
    testcase.assertTrue(path.is_file(), "Phase 7 profile implementation is missing")
    spec = importlib.util.spec_from_file_location("phase7_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Phase7ProfileTests(unittest.TestCase):
    def test_declared_stream_bandwidth_matches_frozen_rates_and_shapes(self):
        contract_path = PHASE7 / "phase7_contract.json"
        self.assertTrue(contract_path.is_file(), "Phase 7 contract is missing")
        contract = json.loads(contract_path.read_text())
        module = load_profile(self)

        result = module.stream_bandwidth(contract["stream_bandwidth"])

        self.assertAlmostEqual(result["native_rgb_mbps"], 18.432)
        self.assertAlmostEqual(result["metric_depth_mbps"], 24.576)
        self.assertAlmostEqual(result["semantic_bev_mbps"], 41.28768)
        self.assertAlmostEqual(result["xfeat_tensor_mbps"], 3.2768)
        self.assertAlmostEqual(result["total_mbps"], 87.57248)

    def test_hardware_selection_chooses_smallest_candidate_with_recommended_headroom(self):
        contract_path = PHASE7 / "phase7_contract.json"
        self.assertTrue(contract_path.is_file(), "Phase 7 contract is missing")
        contract = json.loads(contract_path.read_text())
        module = load_profile(self)

        selected = module.select_hardware(
            contract["hardware_candidates"], contract["recommended_requirements"]
        )

        self.assertEqual(selected["id"], "jetson_orin_nx_16gb_super")

    def test_profile_gate_requires_cuda_and_latency_budget(self):
        module = load_profile(self)
        acceptance = {
            "control_pipeline_p95_ms_max": 50.0,
            "xfeat_p95_ms_max": 500.0,
            "nmpc_p95_ms_max": 10.0,
        }
        healthy = {
            "semantic_provider": "CUDAExecutionProvider",
            "xfeat_provider": "CUDAExecutionProvider",
            "control_pipeline": {"p95_ms": 30.0},
            "xfeat": {"p95_ms": 80.0},
            "nmpc": {"p95_ms": 2.0},
        }

        self.assertTrue(module.profile_gate(healthy, acceptance))
        cpu_only = json.loads(json.dumps(healthy))
        cpu_only["semantic_provider"] = "CPUExecutionProvider"
        self.assertFalse(module.profile_gate(cpu_only, acceptance))
        too_slow = json.loads(json.dumps(healthy))
        too_slow["control_pipeline"]["p95_ms"] = 50.1
        self.assertFalse(module.profile_gate(too_slow, acceptance))

    def test_timing_summary_reports_p50_p95_and_max(self):
        module = load_profile(self)

        result = module.timing_summary([1.0, 2.0, 3.0, 4.0, 100.0])

        self.assertEqual(result["samples"], 5)
        self.assertEqual(result["p50_ms"], 3.0)
        self.assertAlmostEqual(result["p95_ms"], 80.8)
        self.assertEqual(result["max_ms"], 100.0)


if __name__ == "__main__":
    unittest.main()
