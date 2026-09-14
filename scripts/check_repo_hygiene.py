#!/usr/bin/env python3
"""仓库卫生检查 — FSD-SimVerify 验收脚本。

该脚本保证 push 出去的 main 分支始终满足"白盒发布"约束:

* 必需的交付文件存在 (含 HANDOFF、Phase 状态 JSON、active ONNX 模型);
* 不会被发布的目标 (target/, dataset/, artifacts/, contracts/**/out/) 不进入 Git 跟踪;
* active ONNX 模型哈希与 ``model/DELIVERY.json`` 完全一致;
* README 不会再次夹带已经白盒验证失败的旧宣传语。

失败时使用结构化 JSON 输出并以非零退出码退出,便于 CI 与
``scripts/test_repo_hygiene.py`` 直接消费。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "HANDOFF.md",
    "docs/evidence/final_metrics.json",
    "model/warehouse_nav14_candidate.json",
    "model/warehouse_nav14_candidate.onnx",
    "model/xfeat_640x640.onnx",
    "contracts/phase6/phase6_status.json",
    "contracts/phase7/phase7_status.json",
    "contracts/real_vehicle/pre_hardware_status.json",
)
FORBIDDEN_COMPONENTS = {"target", "out", "dataset", "artifacts"}
OBSOLETE_README_CLAIMS = (
    "Spiced Self-Play RL",
    "零样本、零真实数据、一字不改",
    "百元级边缘算力",
)
REPORT_SCHEMA_VERSION = "fsd-simverify-hygiene-v1"


def forbidden_tracked_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    if normalized.startswith("docs/evidence/"):
        return False
    return bool(FORBIDDEN_COMPONENTS.intersection(normalized.split("/")))


def git_lines(extra_args: Iterable[str]) -> tuple[list[str], str | None]:
    try:
        result = subprocess.run(
            ["git", *extra_args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        return [], f"git {' '.join(extra_args)} failed: {detail.strip()}"
    return [line for line in result.stdout.splitlines() if line], None


def is_ignored(path: str) -> bool:
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def model_hash_errors(root: Path, manifest: dict) -> list[str]:
    errors: list[str] = []
    for model in manifest.get("active_models", []):
        path = root / model["path"]
        if not path.is_file():
            errors.append(f"active model is missing: {model['path']}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != model["sha256"]:
            errors.append(f"active model hash mismatch: {model['path']}")
    return errors


@dataclass
class HygieneReport:
    schema_version: str = REPORT_SCHEMA_VERSION
    repo_root: str = str(ROOT)
    ok: bool = False
    required_files: list[dict] = field(default_factory=list)
    forbidden_paths: list[str] = field(default_factory=list)
    ignored_required_files: list[str] = field(default_factory=list)
    obsolete_readme_claims: list[str] = field(default_factory=list)
    model_hash_errors: list[str] = field(default_factory=list)
    other_errors: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "repo_root": self.repo_root,
            "ok": self.ok,
            "summary": {
                "required_files_checked": len(self.required_files),
                "required_files_missing": [
                    entry["path"] for entry in self.required_files if not entry["present"]
                ],
                "forbidden_tracked_paths": self.forbidden_paths,
                "ignored_required_files": self.ignored_required_files,
                "obsolete_readme_claims": self.obsolete_readme_claims,
                "model_hash_errors": self.model_hash_errors,
                "other_errors": self.other_errors,
            },
            "required_files": self.required_files,
        }


def build_report() -> HygieneReport:
    report = HygieneReport()
    for path in REQUIRED_FILES:
        report.required_files.append({"path": path, "present": (ROOT / path).is_file()})
        if not (ROOT / path).is_file():
            report.other_errors.append(f"required delivery file is missing: {path}")

    tracked, git_error = git_lines(["ls-files"])
    if git_error:
        report.other_errors.append(git_error)
    for path in tracked:
        if forbidden_tracked_path(path):
            report.forbidden_paths.append(path)

    for entry in report.required_files:
        path = entry["path"]
        if path.endswith(".onnx") and is_ignored(path):
            report.ignored_required_files.append(path)

    readme_path = ROOT / "README.md"
    if readme_path.is_file():
        readme = readme_path.read_text(encoding="utf-8")
        for claim in OBSOLETE_README_CLAIMS:
            if claim in readme:
                report.obsolete_readme_claims.append(claim)
    else:
        report.other_errors.append("README.md is missing")

    delivery_path = ROOT / "model/DELIVERY.json"
    if delivery_path.is_file():
        delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
        report.model_hash_errors.extend(model_hash_errors(ROOT, delivery))
    else:
        report.other_errors.append("model/DELIVERY.json is missing")

    report.ok = not (
        report.forbidden_paths
        or report.ignored_required_files
        or report.obsolete_readme_claims
        or report.model_hash_errors
        or report.other_errors
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="仅输出结构化 JSON, 便于 CI 解析",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="把结构化报告写入指定路径 (与 --json 组合使用)",
    )
    args = parser.parse_args(argv)
    report = build_report()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report.to_jsonable(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    if args.json:
        json.dump(report.to_jsonable(), sys.stdout, indent=2, ensure_ascii=False, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if report.ok:
            print("Repository hygiene validation OK")
            print(
                f"{len(REQUIRED_FILES)} delivery files present; "
                "active models are publishable"
            )
        else:
            print("Repository hygiene validation FAILED", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
