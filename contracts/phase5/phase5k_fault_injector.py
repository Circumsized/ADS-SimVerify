#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path

import pyarrow as pa

from phase5j_runtime import sigkill_and_confirm
from phase5k_faults import (
    atomic_write_json,
    discover_process_pid,
    enospc_observed,
    episode_reset_due,
    process_table,
    write_enospc_probe,
    watchdog_reset_due,
)


def metadata_value(event, key, default=None):
    metadata = event.get("metadata") or {}
    return metadata.get(key, (metadata.get("parameters") or {}).get(key, default))


def main():
    from dora import Node

    run_mode = os.environ["PHASE5K_RUN_MODE"]
    output = Path(os.environ["PHASE5K_OUTPUT"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    node = Node()
    injected = False
    reset_sent = False
    resource_process = None
    resource_log = None
    resource_reported = False
    previous_episode = None
    last_watchdog_reset_frame = None

    def send_event(frame_id, action, target, confirmed, **extra):
        metadata = {
            "source_frame_id": int(frame_id),
            "action": str(action),
            "target": str(target),
            "confirmed": bool(confirmed),
            **extra,
        }
        try:
            node.send_output("fault_event", pa.array([int(frame_id)], type=pa.int64()), metadata=metadata)
            return True
        except BaseException:
            return False

    def request_stop(frame_id):
        node.send_output(
            "safety_request",
            pa.array([1, 0], type=pa.int8()),
            metadata={"source_frame_id": int(frame_id), "emergency_stop": True},
        )

    def request_reset(frame_id):
        node.send_output(
            "safety_request",
            pa.array([0, 1], type=pa.int8()),
            metadata={"source_frame_id": int(frame_id), "reset": True},
        )

    def inject_process_kill(frame_id, target):
        pid = discover_process_pid(target, process_table())
        receipt_path = output / "fault_receipt.json"
        receipt = {
            "schema_version": "phase5k-process-fault-v1",
            "action": "sigkill",
            "target": target,
            "pid": pid,
            "source_frame_id": int(frame_id),
            "kill_monotonic_ns": time.monotonic_ns(),
            "confirmed": False,
        }
        atomic_write_json(receipt_path, receipt)
        send_event(frame_id, "sigkill", target, False, pid=pid)
        confirmed = sigkill_and_confirm(pid)
        receipt["confirmed"] = confirmed
        receipt["confirmed_monotonic_ns"] = time.monotonic_ns()
        atomic_write_json(receipt_path, receipt)
        send_event(frame_id, "sigkill", target, confirmed, pid=pid)
        if not confirmed:
            raise RuntimeError(f"SIGKILL was not confirmed for {target} pid={pid}")
        if target == "coordinator":
            request_stop(frame_id)

    try:
        while True:
            event = node.next(timeout=0.1)
            if resource_process is not None and resource_process.poll() is not None and not resource_reported:
                resource_reported = True
                if resource_log is not None:
                    resource_log.close()
                    resource_log = None
                receipt_path = output / "gpu_oom_receipt.json"
                receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {"observed": False}
                send_event(
                    receipt.get("source_frame_id", 80),
                    "cuda_oom",
                    "gpu",
                    bool(receipt.get("observed")),
                    returncode=int(resource_process.returncode),
                )
            if event is None:
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            if event["id"] == "run_complete":
                break
            if event["id"] != "plant_telemetry":
                continue
            frame_id = int(metadata_value(event, "source_frame_id"))
            episode_index = int(metadata_value(event, "episode_index", 0))
            episode_changed = run_mode == "hour_endurance" and episode_reset_due(previous_episode, episode_index)
            if episode_changed:
                request_reset(frame_id)
                send_event(frame_id, "episode_reset", "safety_supervisor", True, episode_index=episode_index)
            previous_episode = episode_index
            if not episode_changed and watchdog_reset_due(
                run_mode,
                str(metadata_value(event, "safety_state", "boot")),
                str(metadata_value(event, "safety_reason", "none")),
                frame_id,
                last_watchdog_reset_frame,
            ):
                request_reset(frame_id)
                last_watchdog_reset_frame = frame_id
                send_event(frame_id, "watchdog_reset", "safety_supervisor", True)

            if not injected and frame_id >= 80:
                injected = True
                if run_mode in {"coordinator_sigkill", "daemon_sigkill"}:
                    inject_process_kill(frame_id, run_mode.removesuffix("_sigkill"))
                elif run_mode == "disk_full_recovery":
                    request_stop(frame_id)
                    receipt = write_enospc_probe("/dev/full")
                    receipt.update({
                        "schema_version": "phase5k-disk-full-v1",
                        "source_frame_id": frame_id,
                        "observed": enospc_observed(receipt),
                    })
                    atomic_write_json(output / "disk_full_receipt.json", receipt)
                    send_event(frame_id, "enospc", "disk", receipt["observed"], errno=receipt["errno"] or -1)
                elif run_mode == "gpu_oom_recovery":
                    request_stop(frame_id)
                    script = Path(__file__).with_name("phase5k_gpu_oom_probe.py")
                    receipt = output / "gpu_oom_receipt.json"
                    resource_log = (output / "gpu_oom_probe.log").open("w", encoding="ascii")
                    resource_process = subprocess.Popen(
                        ["/home/zhz/isaacsim/python.sh", str(script), str(receipt)],
                        stdout=resource_log,
                        stderr=subprocess.STDOUT,
                    )

            resource_done = run_mode == "disk_full_recovery" or (
                resource_process is not None and resource_process.poll() is not None
            )
            if (
                not reset_sent
                and frame_id >= 92
                and run_mode in {"gpu_oom_recovery", "disk_full_recovery"}
                and resource_done
            ):
                request_reset(frame_id)
                reset_sent = True
                send_event(frame_id, "reset", "safety_supervisor", True)
    finally:
        if resource_process is not None and resource_process.poll() is None:
            resource_process.terminate()
            try:
                resource_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                resource_process.kill()
                resource_process.wait()
        if resource_log is not None:
            resource_log.close()


if __name__ == "__main__":
    main()
