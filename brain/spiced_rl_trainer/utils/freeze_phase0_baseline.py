#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "baselines"
PAPER_DICTIONARY = Path(os.environ["PAPER_DICTIONARY_PATH"]) if os.environ.get("PAPER_DICTIONARY_PATH") else Path.home() / ".config" / "paper_dictionary.md"


def run(command):
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def optional_run(command):
    try:
        return run(command).strip()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_and_record(source, destination, baseline_dir, records, role):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(
        {
            "path": destination.relative_to(baseline_dir).as_posix(),
            "source": str(source),
            "role": role,
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
    )


def copy_tree(source_dir, destination_dir, baseline_dir, records, role):
    if not source_dir.is_dir():
        return
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        copy_and_record(
            source,
            destination_dir / source.relative_to(source_dir),
            baseline_dir,
            records,
            role,
        )


def dataset_summary(dataset_dir):
    def count_rows(path):
        with path.open("rb") as handle:
            return sum(1 for _ in handle)

    csv_files = sorted(dataset_dir.rglob("*.csv"))
    rows = sum(count_rows(path) for path in csv_files)
    return {
        "csv_files": len(csv_files),
        "csv_rows_including_headers": rows,
        "purified_files": len(list((dataset_dir / "purified").glob("*.csv"))),
        "purified_rows_including_headers": sum(
            count_rows(path) for path in (dataset_dir / "purified").glob("*.csv")
        ),
    }


def write_generated(path, content, baseline_dir, records, role):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    kwargs = {} if mode == "wb" else {"encoding": "utf-8"}
    with path.open(mode, **kwargs) as handle:
        handle.write(content)
    records.append(
        {
            "path": path.relative_to(baseline_dir).as_posix(),
            "source": "generated",
            "role": role,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )


def freeze(output_root, baseline_id):
    baseline_dir = output_root / baseline_id
    baseline_dir.mkdir(parents=True, exist_ok=False)
    records = []

    head = run(["git", "rev-parse", "HEAD"]).strip()
    branch = run(["git", "branch", "--show-current"]).strip()
    status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"])

    archive_path = baseline_dir / "source" / f"tracked_source_{head[:12]}.tar.gz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "archive",
            "--format=tar.gz",
            f"--output={archive_path}",
            head,
        ]
    )
    records.append(
        {
            "path": archive_path.relative_to(baseline_dir).as_posix(),
            "source": f"git:{head}",
            "role": "tracked source snapshot",
            "bytes": archive_path.stat().st_size,
            "sha256": sha256(archive_path),
        }
    )

    write_generated(
        baseline_dir / "source" / "worktree.patch",
        run(["git", "diff", "--binary", "HEAD"]),
        baseline_dir,
        records,
        "tracked worktree changes",
    )
    write_generated(
        baseline_dir / "source" / "git_status.txt",
        status,
        baseline_dir,
        records,
        "worktree status before freeze",
    )

    untracked = run(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    for relative in sorted(untracked):
        if relative == "bc_anchor.pth" or relative.startswith(("model/", "artifacts/")):
            continue
        source = REPO_ROOT / relative
        if source.is_file():
            copy_and_record(
                source,
                baseline_dir / "source" / "untracked" / relative,
                baseline_dir,
                records,
                "untracked worktree file",
            )

    copy_tree(REPO_ROOT / "model", baseline_dir / "model", baseline_dir, records, "model artifact")
    copy_tree(REPO_ROOT / "dataset", baseline_dir / "dataset", baseline_dir, records, "dataset artifact")
    for relative, role in [
        (Path("bc_anchor.pth"), "BC anchor weights"),
        (Path("assets/fsd_car_racetrack.usd"), "simulation scene"),
    ]:
        source = REPO_ROOT / relative
        if source.is_file():
            copy_and_record(source, baseline_dir / relative, baseline_dir, records, role)
    if PAPER_DICTIONARY.is_file():
        copy_and_record(
            PAPER_DICTIONARY,
            baseline_dir / "references" / "paper_dictionary.md",
            baseline_dir,
            records,
            "paper audit baseline",
        )

    environment = {
        "uname": optional_run(["uname", "-a"]),
        "os_release": Path("/etc/os-release").read_text(encoding="utf-8", errors="replace"),
        "cpu": optional_run(["lscpu"]),
        "memory": Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace"),
        "gpu": optional_run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "python": optional_run([sys.executable, "--version"]),
        "isaac_python": optional_run([os.environ.get("ISAAC_SIM_PYTHON", os.environ.get("ISAAC_SIM_PYTHON_SH", "isaacsim/python.sh")), "--version"]),
        "uv": optional_run(["uv", "--version"]),
        "rustc": optional_run(["rustc", "--version"]),
        "cargo": optional_run(["cargo", "--version"]),
        "pnpm": optional_run(["pnpm", "--version"]),
        "dora": optional_run(["dora", "--version"]),
    }
    write_generated(
        baseline_dir / "environment.json",
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        baseline_dir,
        records,
        "hardware and toolchain inventory",
    )

    manifest = {
        "schema_version": 1,
        "baseline_id": baseline_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Phase 0 reproducible baseline before perception-policy-control refactor",
        "model_status": {
            "model/spiced_brain.onnx": (
                "Guarded BC baseline. No PPO evaluation checkpoint exceeded the BC reward 6.68 "
                "during the 2026-07-15 run."
            ),
            "model/pidnet_s.onnx": "PIDNet-S Cityscapes 19-class perception baseline",
        },
        "git": {
            "head": head,
            "branch": branch,
            "dirty_before_freeze": bool(status.strip()),
        },
        "dataset": dataset_summary(REPO_ROOT / "dataset"),
        "files": sorted(records, key=lambda item: item["path"]),
    }
    manifest_path = baseline_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_lines = [f'{item["sha256"]}  {item["path"]}' for item in manifest["files"]]
    (baseline_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    for path in sorted(baseline_dir.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    baseline_dir.chmod(0o555)
    return baseline_dir


def verify(baseline_dir):
    manifest_path = baseline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = baseline_dir / item["path"]
        if not path.is_file():
            failures.append(f'missing: {item["path"]}')
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != item["bytes"] or actual_hash != item["sha256"]:
            failures.append(
                f'mismatch: {item["path"]} size={actual_size} sha256={actual_hash}'
            )
    if failures:
        print("Baseline verification FAILED")
        print("\n".join(failures))
        return 1
    print(
        f'Baseline verification OK: {manifest["baseline_id"]} '
        f'({len(manifest["files"])} files)'
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description="Create or verify the Phase 0 baseline snapshot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    create_parser.add_argument("--baseline-id")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("baseline_dir", type=Path)
    args = parser.parse_args()

    if args.command == "create":
        baseline_id = args.baseline_id or datetime.now().strftime("phase0_%Y%m%d_%H%M%S")
        baseline_dir = freeze(args.output_root.resolve(), baseline_id)
        print(f"Phase 0 baseline created: {baseline_dir}")
        return 0
    return verify(args.baseline_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
