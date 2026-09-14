#!/usr/bin/env python3
import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PHASE5 = ROOT / "contracts/phase5"


class Phase5JTopologyTests(unittest.TestCase):
    def test_contract_freezes_fault_modes_and_endurance_scope(self):
        contract = json.loads((PHASE5 / "phase5j_contract.json").read_text())

        self.assertEqual(contract["parent_status"], "contracts/phase5/phase5i_status.json")
        self.assertEqual(contract["endurance"]["scenario_cycles"], 2)
        self.assertGreaterEqual(contract["endurance"]["minimum_frames"], 4000)
        self.assertEqual(
            contract["fault_gate"]["runs"],
            ["controller_sigkill", "supervisor_sigkill", "sensor_freeze", "restart_recovery"],
        )
        self.assertLessEqual(contract["fault_gate"]["stop_latency_frames_max"], 3)
        self.assertFalse(contract["promotion"]["real_vehicle_control_allowed"])

    def test_topology_separates_control_process_from_isaac_actuator(self):
        topology = yaml.safe_load((PHASE5 / "dora_dataflow_phase5j_resilience.yaml").read_text())
        nodes = {node["id"]: node for node in topology["nodes"]}

        self.assertEqual(
            set(nodes),
            {
                "phase5j_isaac_plant",
                "phase5j_controller",
                "phase5j_safety_supervisor",
                "phase5j_fault_injector",
                "phase5j_evidence_sink",
            },
        )
        plant = nodes["phase5j_isaac_plant"]
        controller = nodes["phase5j_controller"]
        supervisor = nodes["phase5j_safety_supervisor"]
        injector = nodes["phase5j_fault_injector"]
        self.assertEqual(plant["inputs"]["safe_control"]["source"], "phase5j_safety_supervisor/safe_control")
        self.assertEqual(supervisor["inputs"]["proposed_control"]["source"], "phase5j_controller/proposed_control")
        self.assertIn("tick", supervisor["inputs"])
        self.assertNotIn("safe_control", controller["outputs"])
        self.assertNotIn("safe_control", injector["outputs"])
        self.assertEqual(injector["inputs"]["controller_heartbeat"]["source"], "phase5j_controller/controller_heartbeat")
        self.assertEqual(injector["inputs"]["supervisor_heartbeat"]["source"], "phase5j_safety_supervisor/supervisor_heartbeat")

    def test_all_node_entrypoints_exist(self):
        topology = yaml.safe_load((PHASE5 / "dora_dataflow_phase5j_resilience.yaml").read_text())
        for node in topology["nodes"]:
            self.assertTrue((PHASE5 / node["args"]).is_file(), node["args"])


if __name__ == "__main__":
    unittest.main()
