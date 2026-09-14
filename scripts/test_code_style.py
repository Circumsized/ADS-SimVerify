import tempfile
import unittest
from pathlib import Path

import check_code_style


class CodeStyleCheckerTests(unittest.TestCase):
    def test_scope_excludes_frozen_and_documentation_directories(self):
        root = check_code_style.ROOT
        self.assertFalse(check_code_style.in_scope(root / "contracts/x.py"))
        self.assertFalse(check_code_style.in_scope(root / "assets/x.py"))
        self.assertTrue(check_code_style.in_scope(root / "scripts/check_code_style.py"))

    def test_code_cjk_is_checked_but_comments_are_allowed(self):
        self.assertNotIn("中文", check_code_style.code_without_strings_and_comments("x = '中文'\n# 中文", ".py"))
        self.assertIn("中文", check_code_style.code_without_strings_and_comments("x = 中文", ".py"))

    def test_dataflow_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dora.yaml"
            path.write_text("- id: isaac_sim_env\n", encoding="utf-8")
            self.assertTrue(check_code_style.check_dataflow(path))

    def test_real_repository_passes_style_gate(self):
        errors = check_code_style.check_files()
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
