#!/usr/bin/env python3
import errno
import json
import mmap
import os
import shlex
import struct
from dataclasses import dataclass
from pathlib import Path


RUN_MODES = {
    "coordinator_sigkill",
    "daemon_sigkill",
    "gpu_oom_recovery",
    "disk_full_recovery",
    "hour_endurance",
}


def phase5j_run_mode(run_mode):
    if run_mode not in RUN_MODES:
        raise ValueError(f"unsupported run mode: {run_mode}")
    return "endurance" if run_mode == "hour_endurance" else run_mode


def episode_reset_due(previous_episode, current_episode):
    return previous_episode is not None and int(current_episode) != int(previous_episode)


def watchdog_reset_due(run_mode, safety_state, safety_reason, frame_id, last_reset_frame, cooldown_frames=4):
    if (
        run_mode != "hour_endurance"
        or safety_state != "fault"
        or safety_reason != "missing_runtime_input"
    ):
        return False
    return last_reset_frame is None or int(frame_id) - int(last_reset_frame) >= int(cooldown_frames)


@dataclass(frozen=True)
class FaultAction:
    kill_target: str | None = None
    gpu_oom: bool = False
    disk_full: bool = False
    reset: bool = False

    @property
    def has_action(self):
        return self.kill_target is not None or self.gpu_oom or self.disk_full or self.reset


def fault_action(run_mode, frame_id):
    if run_mode not in RUN_MODES:
        raise ValueError(f"unsupported run mode: {run_mode}")
    frame_id = int(frame_id)
    if run_mode == "coordinator_sigkill" and frame_id == 80:
        return FaultAction(kill_target="coordinator")
    if run_mode == "daemon_sigkill" and frame_id == 80:
        return FaultAction(kill_target="daemon")
    if run_mode == "gpu_oom_recovery":
        if frame_id == 80:
            return FaultAction(gpu_oom=True)
        if frame_id == 92:
            return FaultAction(reset=True)
    if run_mode == "disk_full_recovery":
        if frame_id == 80:
            return FaultAction(disk_full=True)
        if frame_id == 92:
            return FaultAction(reset=True)
    return FaultAction()


def discover_process_pid(role, process_table):
    if role not in {"coordinator", "daemon"}:
        raise ValueError(f"unsupported Dora role: {role}")
    matches = []
    for pid, command in process_table:
        tokens = shlex.split(command)
        for index, token in enumerate(tokens[:-1]):
            if Path(token).name == "dora" and tokens[index + 1] == role:
                matches.append(int(pid))
                break
    if len(matches) != 1:
        raise RuntimeError(f"expected one dora {role}, found {matches}")
    return matches[0]


def process_table():
    rows = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command:
            rows.append((int(entry.name), command))
    return rows


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    os.replace(temporary, path)


def write_enospc_probe(path):
    try:
        Path(path).write_bytes(b"phase5k-enospc-probe")
    except OSError as error:
        return {"written": False, "errno": error.errno, "message": str(error)}
    return {"written": True, "errno": None, "message": "written"}


class SafetyLedger:
    def __init__(self, path, size=4096):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = int(size)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        os.posix_fallocate(self.fd, 0, self.size)
        self.mapping = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_WRITE)
        self.update(-1, 0, 0.0, 0.0, "initialized", os.getpid())

    def update(self, sequence_id, monotonic_ns, linear, angular, reason, pid):
        payload = json.dumps(
            {
                "schema_version": "phase5k-safety-ledger-v1",
                "sequence_id": int(sequence_id),
                "monotonic_ns": int(monotonic_ns),
                "linear": float(linear),
                "angular": float(angular),
                "reason": str(reason),
                "pid": int(pid),
            },
            separators=(",", ":"),
        ).encode("ascii")
        if len(payload) + 4 > self.size:
            raise ValueError("safety ledger payload exceeds reserved size")
        self.mapping.seek(0)
        self.mapping.write(struct.pack("<I", len(payload)))
        self.mapping.write(payload)
        if "fail_safe" in str(reason) or "watchdog" in str(reason) or "terminal" in str(reason):
            self.mapping.flush()

    def close(self):
        if self.mapping is not None:
            self.mapping.flush()
            self.mapping.close()
            self.mapping = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def read(path):
        with Path(path).open("rb") as source:
            size_bytes = source.read(4)
            if len(size_bytes) != 4:
                raise ValueError("safety ledger header is incomplete")
            size = struct.unpack("<I", size_bytes)[0]
            return json.loads(source.read(size).decode("ascii"))


def enospc_observed(result):
    return not result["written"] and result["errno"] == errno.ENOSPC


def host_watchdog_due(receipt, ledger):
    return bool(
        receipt.get("target") == "daemon"
        and receipt.get("confirmed")
        and (float(ledger.get("linear", 0.0)) != 0.0 or float(ledger.get("angular", 0.0)) != 0.0)
    )
