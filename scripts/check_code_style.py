#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"contracts", "assets", "docs", ".git"}
EXEMPT = {"scripts/check_repo_hygiene.py", "scripts/test_repo_hygiene.py"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
FORBIDDEN = ("[" + "cite:", "架" + "构师", "协议" + "确认", "自" + "愈", "NEX" + "US", "SO" + "TA", "核" + "弹", "工业" + "级")
PATH_TOKENS = ("/home/" + "zhz", "/run/media/" + "zhz")
NODE_IDS = {"isaac_sim_env", "core_perception", "optical_reflex_node", "fast_brain_nmpc", "slow_brain_mapper", "telemetry_dashboard"}
PORTS = {"control_cmd", "xfeat_features", "odometry", "jpeg_image", "bev_grid", "bev_semantic", "ttc", "human_prior"}


def in_scope(path):
    relative = path.relative_to(ROOT).as_posix()
    if relative in EXEMPT or any(part in EXCLUDED for part in path.relative_to(ROOT).parts):
        return False
    return (relative.startswith("core-") and path.suffix == ".rs") or (relative.startswith(("simulation-env/", "brain/", "scripts/")) and path.suffix == ".py") or (path.name.startswith("dora_dataflow") and path.suffix == ".yaml")


def code_without_strings_and_comments(text, suffix):
    if suffix == ".py":
        import tokenize
        from io import StringIO
        kept = []
        for token in tokenize.generate_tokens(StringIO(text).readline):
            if token.type not in (tokenize.STRING, tokenize.COMMENT):
                kept.append(token.string)
        return "\n".join(kept)
    if suffix in (".yaml", ".yml"):
        return re.sub(r"(?:^|\s)#.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', "", text)


def check_files(root=ROOT):
    errors = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not in_scope(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        code = code_without_strings_and_comments(text, path.suffix)
        for label, pattern in (("CJK in code", CJK), ("emoji", EMOJI)):
            source_for_emoji = text.replace("✓", "")
            if pattern.search(code if label.startswith("CJK") else source_for_emoji):
                errors.append(f"{label}: {path.relative_to(root)}")
        for token in FORBIDDEN + PATH_TOKENS:
            if token in text:
                errors.append(f"forbidden token {token}: {path.relative_to(root)}")
    errors.extend(check_dataflow(root / "dora_dataflow.yaml"))
    return errors


def check_dataflow(path):
    text = path.read_text(encoding="utf-8")
    ids = set(re.findall(r"^\s*- id:\s*([^\s#]+)", text, re.M))
    ports = set(re.findall(r"^\s{6}- ([a-z][a-z0-9_]*)\s*$", text, re.M))
    ports.update(re.findall(r"^\s{6}([a-z][a-z0-9_]*):\s*[^\n]+", text, re.M))
    errors = []
    if ids != NODE_IDS:
        errors.append(f"dataflow node ids mismatch: {sorted(ids)}")
    if not PORTS.issubset(ports):
        errors.append(f"dataflow ports missing: {sorted(PORTS - ports)}")
    return errors


def main():
    errors = check_files()
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    print("Code style validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
