#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE7 = ROOT / "contracts/phase7"


class Phase7ValidationTests(unittest.TestCase):
    def load_validator(self):
        path = PHASE7 / "validate_phase7.py"
        self.assertTrue(path.is_file(), "Phase 7 validator is missing")
        spec = importlib.util.spec_from_file_location("validate_phase7", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_status_references_parent_contract_implementation_profile_and_report(self):
        module = self.load_validator()
        status = {
            "phase6_status": {"path": "phase6.json", "sha256": "a"},
            "contract": {"path": "contract.json", "sha256": "b"},
            "implementation": {"profiler": {"path": "profiler.py", "sha256": "c"}},
            "profile_evidence": {"path": "profile.json", "sha256": "d"},
            "deployment_report": {"path": "report.md", "sha256": "e"},
        }

        references = module.status_references(status)

        self.assertEqual(
            [item["path"] for item in references],
            ["phase6.json", "contract.json", "profiler.py", "profile.json", "report.md"],
        )

    def test_recommendation_compares_selected_candidate_only(self):
        module = self.load_validator()
        selected = {"id": "jetson_orin_nx_16gb_super", "name": "Orin NX 16GB"}
        profile = {"recommendation": {"selected": selected, "minimum_prototype": {}}}

        self.assertTrue(module.recommendation_matches(profile, selected))
        self.assertFalse(module.recommendation_matches(profile, {"id": "other"}))


if __name__ == "__main__":
    unittest.main()
