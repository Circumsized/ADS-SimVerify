#!/usr/bin/env python3
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_repo_hygiene.py"


def load_checker(testcase):
    testcase.assertTrue(CHECKER.is_file(), "repository hygiene checker is missing")
    spec = importlib.util.spec_from_file_location("check_repo_hygiene", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class RepositoryHygieneTests(unittest.TestCase):
    def test_generated_or_private_paths_must_not_be_tracked(self):
        module = load_checker(self)

        self.assertTrue(module.forbidden_tracked_path("target/debug/app"))
        self.assertTrue(module.forbidden_tracked_path("dataset/spice_run.csv"))
        self.assertTrue(module.forbidden_tracked_path("artifacts/raw/frames.csv"))
        self.assertTrue(module.forbidden_tracked_path("contracts/phase5/out/run/log.txt"))
        self.assertTrue(module.forbidden_tracked_path("contracts/phase5/artifacts/smoke.json"))
        self.assertFalse(module.forbidden_tracked_path("contracts/phase7/phase7_status.json"))
        self.assertFalse(module.forbidden_tracked_path("docs/evidence/final_metrics.json"))

    def test_delivery_requires_active_models_and_handoff_but_not_inactive_models(self):
        module = load_checker(self)

        self.assertIn("model/warehouse_nav14_candidate.onnx", module.REQUIRED_FILES)
        self.assertIn("model/xfeat_640x640.onnx", module.REQUIRED_FILES)
        self.assertIn("HANDOFF.md", module.REQUIRED_FILES)
        self.assertNotIn("model/spiced_brain.onnx", module.REQUIRED_FILES)
        self.assertNotIn("model/pidnet_s.onnx", module.REQUIRED_FILES)

    def test_model_delivery_hash_detects_corrupted_weight(self):
        module = load_checker(self)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"valid model")
            manifest = {
                "active_models": [
                    {
                        "path": "model.onnx",
                        "sha256": hashlib.sha256(b"valid model").hexdigest(),
                    }
                ]
            }

            self.assertEqual(module.model_hash_errors(root, manifest), [])
            model.write_bytes(b"corrupted")
            self.assertTrue(module.model_hash_errors(root, manifest))

    def test_build_report_marks_missing_required_files(self):
        module = load_checker(self)
        with tempfile.TemporaryDirectory() as directory:
            original_root = module.ROOT
            module.ROOT = Path(directory)
            try:
                module.REQUIRED_FILES = ("must_exist.json",)
                report = module.build_report()
            finally:
                module.ROOT = original_root

        self.assertFalse(report.ok)
        self.assertTrue(report.required_files)
        self.assertIn(
            "must_exist.json",
            report.to_jsonable()["summary"]["required_files_missing"],
        )

    def test_json_mode_emits_valid_report(self):
        module = load_checker(self)
        with tempfile.TemporaryDirectory() as directory:
            original_root = module.ROOT
            module.ROOT = Path(directory)
            try:
                # No git, no models — expect an "ok=False" report without exceptions.
                buffer = io.StringIO()
                import contextlib

                with contextlib.redirect_stdout(buffer):
                    exit_code = module.main(["--json"])
            finally:
                module.ROOT = original_root

        self.assertEqual(exit_code, 1)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["schema_version"], module.REPORT_SCHEMA_VERSION)
        self.assertFalse(payload["ok"])
        self.assertIn("model/DELIVERY.json is missing", payload["summary"]["other_errors"])


if __name__ == "__main__":
    unittest.main()