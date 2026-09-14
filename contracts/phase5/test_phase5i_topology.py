#!/usr/bin/env python3
import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PHASE5 = ROOT / "contracts/phase5"


class Phase5ITopologyTests(unittest.TestCase):
    def test_contract_freezes_dual_resolution_without_downscaling_xfeat_source(self):
        contract = json.loads((PHASE5 / "phase5i_contract.json").read_text())

        self.assertEqual(contract["sensor_pipeline"]["source_resolution"], [640, 480])
        self.assertEqual(contract["sensor_pipeline"]["xfeat_input_resolution"], [640, 480])
        self.assertEqual(contract["sensor_pipeline"]["semantic_control_resolution"], [320, 240])
        self.assertFalse(contract["sensor_pipeline"]["xfeat_on_control_critical_path"])
        self.assertEqual(contract["timing"]["control_rate_hz"], 20)
        self.assertEqual(contract["timing"]["sensor_to_wheel_p95_ms_max"], 50.0)

    def test_dora_topology_routes_every_wheel_command_through_safety_supervisor(self):
        topology = yaml.safe_load((PHASE5 / "dora_dataflow_phase5i_control.yaml").read_text())
        nodes = {node["id"]: node for node in topology["nodes"]}

        self.assertEqual(
            set(nodes),
            {
                "phase5i_articulation_runtime",
                "phase5i_safety_supervisor",
                "phase5i_operator_safety",
                "phase5i_evidence_sink",
            },
        )
        runtime = nodes["phase5i_articulation_runtime"]
        supervisor = nodes["phase5i_safety_supervisor"]
        operator = nodes["phase5i_operator_safety"]
        sink = nodes["phase5i_evidence_sink"]
        self.assertEqual(
            runtime["inputs"]["safe_control"]["source"],
            "phase5i_safety_supervisor/safe_control",
        )
        self.assertEqual(
            supervisor["inputs"]["proposed_control"]["source"],
            "phase5i_articulation_runtime/proposed_control",
        )
        self.assertEqual(
            supervisor["inputs"]["safety_request"]["source"],
            "phase5i_operator_safety/safety_request",
        )
        self.assertIn("safety_request", operator["outputs"])
        self.assertEqual(
            sink["inputs"]["articulation_telemetry"]["source"],
            "phase5i_articulation_runtime/articulation_telemetry",
        )
        self.assertEqual(
            sink["inputs"]["localization_image_640"]["source"],
            "phase5i_articulation_runtime/localization_image_640",
        )
        self.assertNotIn("control_cmd", runtime["outputs"])
        self.assertIn("run_complete", runtime["outputs"])
        self.assertEqual(
            supervisor["inputs"]["run_complete"]["source"],
            "phase5i_articulation_runtime/run_complete",
        )
        self.assertEqual(
            operator["inputs"]["run_complete"]["source"],
            "phase5i_articulation_runtime/run_complete",
        )

    def test_all_topology_node_entrypoints_exist(self):
        topology = yaml.safe_load((PHASE5 / "dora_dataflow_phase5i_control.yaml").read_text())
        for node in topology["nodes"]:
            entrypoint = PHASE5 / node["args"]
            self.assertTrue(entrypoint.is_file(), f"missing Dora node: {entrypoint}")


if __name__ == "__main__":
    unittest.main()
