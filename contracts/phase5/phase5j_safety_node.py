#!/usr/bin/env python3
import os
import time
import json
from pathlib import Path

import pyarrow as pa

from phase5i_safety import ControlProposal, RuntimeHealth
from phase5j_safety_relay import SafetyRelay


def metadata_value(event, key):
    metadata = event.get("metadata") or {}
    return metadata.get(key, (metadata.get("parameters") or {}).get(key))


def main():
    from dora import Node

    generation = int(os.environ["PHASE5J_GENERATION"])
    contract = json.loads((Path(__file__).with_name("phase5j_contract.json")).read_text())
    relay = SafetyRelay(
        warmup_frames=contract["timing"]["startup_warmup_frames"],
        watchdog_ms=contract["timing"]["supervisor_watchdog_ms"],
    )
    node = Node()
    pid = os.getpid()
    command_sequence = 0

    def publish(output):
        nonlocal command_sequence
        command_sequence += 1
        metadata = {
            "command_sequence_id": command_sequence,
            "source_frame_id": output.source_frame_id,
            "controller_generation": generation,
            "safety_state": output.decision.state.value,
            "safety_reason": output.decision.reason,
            "reset_applied": output.reset_applied,
            "proposal_fresh": output.proposal_fresh,
            "sensor_started_ns": output.sensor_started_ns,
            "supervisor_timestamp_ms": time.monotonic_ns() / 1e6,
        }
        node.send_output(
            "safe_control",
            pa.array([output.decision.linear, output.decision.angular], type=pa.float32()),
            metadata=metadata,
        )
        node.send_output(
            "safety_state",
            pa.array([command_sequence, output.decision.linear, output.decision.angular], type=pa.float32()),
            metadata=metadata,
        )
        node.send_output(
            "supervisor_heartbeat",
            pa.array([pid, generation], type=pa.int64()),
            metadata={"pid": pid, "generation": generation, "source_frame_id": output.source_frame_id},
        )

    while True:
        event = node.next(timeout=0.2)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "run_complete":
            break
        if event["id"] == "safety_request":
            values = event["value"].to_numpy()
            relay.set_emergency_stop(bool(values[0]))
            if bool(values[1]):
                relay.request_reset()
            continue
        if event["id"] == "tick":
            publish(relay.tick(time.monotonic_ns() / 1e6))
            continue
        if event["id"] != "proposed_control":
            continue
        if int(metadata_value(event, "controller_generation")) != generation:
            continue
        frame_id = int(metadata_value(event, "source_frame_id"))
        values = event["value"].to_numpy()
        health = RuntimeHealth(
            frame_id=frame_id,
            timestamp_ms=float(metadata_value(event, "health_timestamp_ms")),
            sensor_valid=bool(metadata_value(event, "sensor_valid")),
            perception_valid=bool(metadata_value(event, "perception_valid")),
            solver_valid=bool(metadata_value(event, "solver_valid")),
            articulation_ready=bool(metadata_value(event, "articulation_ready")),
        )
        proposal = ControlProposal(
            frame_id=frame_id,
            timestamp_ms=float(metadata_value(event, "proposal_timestamp_ms")),
            linear=float(values[0]),
            angular=float(values[1]),
        )
        publish(
            relay.update(
                time.monotonic_ns() / 1e6,
                health,
                proposal,
                int(metadata_value(event, "sensor_started_ns")),
            )
        )


if __name__ == "__main__":
    main()
