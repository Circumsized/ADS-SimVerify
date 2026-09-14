#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE6 = ROOT / "contracts/phase6"


class Phase6ValidationTests(unittest.TestCase):
    def test_matrix_evidence_references_include_image_and_every_case_telemetry(self):
        path = PHASE6 / "validate_phase6.py"
        self.assertTrue(path.is_file(), "Phase 6 validator is missing")
        spec = importlib.util.spec_from_file_location("validate_phase6", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(
            hasattr(module, "matrix_evidence_references"),
            "Phase 6 validator must verify image and telemetry hashes",
        )
        with tempfile.TemporaryDirectory() as temp:
            summary_path = Path(temp) / "summary.json"
            summary = {
                "evidence": "evidence.png",
                "evidence_sha256": "image-hash",
                "cases": [
                    {"telemetry": "a/run.csv", "telemetry_sha256": "a-hash"},
                    {"telemetry": "b/run.csv", "telemetry_sha256": "b-hash"},
                ],
            }
            summary_path.write_text(json.dumps(summary))

            references = module.matrix_evidence_references(summary_path, summary)

        self.assertEqual(
            [(item[0].name, item[1]) for item in references],
            [("evidence.png", "image-hash"), ("run.csv", "a-hash"), ("run.csv", "b-hash")],
        )

    def test_validator_requires_frozen_parent_matrix_and_endurance_hashes(self):
        path = PHASE6 / "validate_phase6.py"
        self.assertTrue(path.is_file(), "Phase 6 validator is missing")
        spec = importlib.util.spec_from_file_location("validate_phase6", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        status = {
            "phase5k_status": {"path": "parent.json", "sha256": "a"},
            "contract": {"path": "contract.json", "sha256": "b"},
            "implementation": {"runner": {"path": "runner.py", "sha256": "c"}},
            "matrix_evidence": {"path": "summary.json", "sha256": "d"},
        }

        references = module.status_references(status)

        self.assertEqual(
            [reference["path"] for reference in references],
            ["parent.json", "contract.json", "runner.py", "summary.json"],
        )


if __name__ == "__main__":
    unittest.main()
