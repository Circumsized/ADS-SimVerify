#!/usr/bin/env python3
import time

from phase5d_runtime_node import metadata_value
from phase5i_safety import ControlProposal, RuntimeHealth, SafetyState, SafetySupervisor


STATE_CODE = {state: index for index, state in enumerate(SafetyState)}


def main():
    import pyarrow as pa
    from dora import Node

    supervisor = SafetySupervisor(warmup_frames=5, watchdog_ms=150)
    emergency_stop = False
    reset_pending = False
    node = Node()
    while True:
        event = node.next(timeout=0.1)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT":
            continue
        if event["id"] == "run_complete":
            break
        if event["id"] == "safety_request":
            request = event["value"].to_numpy()
            if request.size >= 2:
                emergency_stop = bool(request[0])
                reset_pending |= bool(request[1])
            continue
        if event["id"] != "proposed_control":
            continue
        value = event["value"].to_numpy()
        frame_id = int(metadata_value(event, "source_frame_id"))
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
            linear=float(value[0]),
            angular=float(value[1]),
        )
        previous_state = supervisor.state
        reset_applied = reset_pending and previous_state in (
            SafetyState.EMERGENCY_STOP,
            SafetyState.FAULT,
        )
        decision = supervisor.step(
            time.monotonic_ns() / 1e6,
            health,
            proposal,
            emergency_stop=emergency_stop,
            reset=reset_pending,
        )
        reset_pending = False
        metadata = {
            "source_frame_id": frame_id,
            "safety_state": decision.state.value,
            "safety_reason": decision.reason,
            "safety_latched": decision.latched,
            "reset_applied": reset_applied,
            "supervisor_timestamp_ms": time.monotonic_ns() / 1e6,
        }
        node.send_output(
            "safe_control",
            pa.array([decision.linear, decision.angular], type=pa.float32()),
            metadata=metadata,
        )
        node.send_output(
            "safety_state",
            pa.array(
                [frame_id, STATE_CODE[decision.state], decision.linear, decision.angular],
                type=pa.float32(),
            ),
            metadata=metadata,
        )


if __name__ == "__main__":
    main()
