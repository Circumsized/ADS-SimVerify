#!/usr/bin/env python3
import os

import pyarrow as pa

from phase5j_faults import fault_action
from phase5j_runtime import sigkill_and_confirm


def metadata_value(event, key, default=None):
    metadata = event.get("metadata") or {}
    return metadata.get(key, (metadata.get("parameters") or {}).get(key, default))


def main():
    from dora import Node

    run_mode = os.environ["PHASE5J_RUN_MODE"]
    node = Node()
    pids = {}
    applied = set()
    while True:
        event = node.next(timeout=1.0)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "run_complete":
            break
        if event["id"] == "controller_heartbeat":
            pids["controller"] = int(metadata_value(event, "pid"))
            continue
        if event["id"] == "supervisor_heartbeat":
            pids["supervisor"] = int(metadata_value(event, "pid"))
            continue
        if event["id"] != "plant_telemetry":
            continue
        frame_id = int(metadata_value(event, "source_frame_id"))
        action = fault_action(run_mode, frame_id)
        key = (frame_id, action.kill_target, action.freeze_sensor, action.reset)
        if not action.has_action or key in applied:
            continue
        applied.add(key)
        if action.freeze_sensor is not None:
            node.send_output(
                "fault_command",
                pa.array([int(action.freeze_sensor)], type=pa.int8()),
                metadata={"source_frame_id": frame_id, "freeze_sensor": action.freeze_sensor},
            )
        if action.reset:
            node.send_output(
                "safety_request",
                pa.array([0, 1], type=pa.int8()),
                metadata={"source_frame_id": frame_id, "reset": True},
            )
        if action.kill_target is not None:
            pid = pids.get(action.kill_target)
            if pid is None:
                raise RuntimeError(f"missing heartbeat PID for {action.kill_target}")
            node.send_output(
                "fault_event",
                pa.array([pid], type=pa.int64()),
                metadata={
                    "source_frame_id": frame_id,
                    "action": "sigkill",
                    "target": action.kill_target,
                    "pid": pid,
                    "confirmed": False,
                },
            )
            confirmed = sigkill_and_confirm(pid)
            node.send_output(
                "fault_event",
                pa.array([pid], type=pa.int64()),
                metadata={
                    "source_frame_id": frame_id,
                    "action": "sigkill",
                    "target": action.kill_target,
                    "pid": pid,
                    "confirmed": confirmed,
                },
            )
            if not confirmed:
                raise RuntimeError(f"SIGKILL was not confirmed for {action.kill_target} pid={pid}")


if __name__ == "__main__":
    main()
