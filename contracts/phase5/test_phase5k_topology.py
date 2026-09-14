#!/usr/bin/env python3
import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PHASE5 = ROOT / "contracts/phase5"


class Phase5KTopologyTests(unittest.TestCase):
    def test_contract_freezes_real_hour_and_fault_gates(self):
        contract = json.loads((PHASE5 / "phase5k_contract.json").read_text())

        self.assertEqual(contract["parent_status"], "contracts/phase5/phase5j_status.json")
        self.assertEqual(contract["hour_endurance"]["frames"], 72000)
        self.assertEqual(contract["hour_endurance"]["duration_s"], 3600.0)
        self.assertEqual(contract["hour_endurance"]["active_ratio_min"], 0.98)
        self.assertEqual(
            contract["fault_gate"]["runs"],
            ["coordinator_sigkill", "daemon_sigkill", "gpu_oom_recovery", "disk_full_recovery"],
        )
        self.assertEqual(contract["fault_gate"]["reset_not_before_frame"], 92)
        self.assertTrue(contract["fault_gate"]["resource_release_required_before_reset"])
        self.assertEqual(
            contract["fault_gate"]["coordinator_loss_policy"],
            "local_data_plane_fail_safe_stop_then_full_stack_restart",
        )
        self.assertEqual(
            contract["fault_gate"]["daemon_loss_policy"],
            "host_out_of_band_signal_stops_actual_articulation",
        )
        self.assertEqual(contract["hour_endurance"]["missing_input_reset_cooldown_frames"], 4)
        self.assertFalse(contract["promotion"]["real_vehicle_control_allowed"])

    def test_topology_keeps_phase5j_controller_and_supervisor_boundaries(self):
        topology = yaml.safe_load((PHASE5 / "dora_dataflow_phase5k_resilience.yaml").read_text())
        nodes = {node["id"]: node for node in topology["nodes"]}

        self.assertEqual(
            set(nodes),
            {
                "phase5k_isaac_plant",
                "phase5k_controller",
                "phase5k_safety_supervisor",
                "phase5k_fault_injector",
                "phase5k_evidence_sink",
            },
        )
        self.assertEqual(nodes["phase5k_controller"]["args"], "phase5j_controller.py")
        self.assertEqual(nodes["phase5k_safety_supervisor"]["args"], "phase5j_safety_node.py")
        self.assertEqual(
            nodes["phase5k_isaac_plant"]["inputs"]["safe_control"]["source"],
            "phase5k_safety_supervisor/safe_control",
        )
        self.assertNotIn("safe_control", nodes["phase5k_controller"]["outputs"])
        self.assertNotIn("safe_control", nodes["phase5k_fault_injector"]["outputs"])
        self.assertNotIn("phase5k_host_watchdog", nodes)

    def test_all_node_entrypoints_exist(self):
        topology = yaml.safe_load((PHASE5 / "dora_dataflow_phase5k_resilience.yaml").read_text())
        for node in topology["nodes"]:
            self.assertTrue((PHASE5 / node["args"]).is_file(), node["args"])


if __name__ == "__main__":
    unittest.main()
