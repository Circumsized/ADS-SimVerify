#!/usr/bin/env python3
import os


RUN_MODES = {"nominal", "emergency_stop_reset", "watchdog_reset"}


def safety_request_for(run_mode, tick):
    if run_mode == "emergency_stop_reset":
        if 40 <= tick <= 49:
            return True, False
        if tick == 50:
            return False, True
    if run_mode == "watchdog_reset" and tick == 65:
        return False, True
    return False, False


def main():
    import pyarrow as pa
    from dora import Node

    run_mode = os.environ.get("PHASE5I_RUN_MODE", "nominal")
    if run_mode not in RUN_MODES:
        raise SystemExit(f"unsupported PHASE5I_RUN_MODE: {run_mode}")
    node = Node()
    tick = 0
    while True:
        event = node.next(timeout=1.0)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] == "INPUT" and event["id"] == "run_complete":
            break
        if event["type"] != "INPUT" or event["id"] != "tick":
            continue
        emergency_stop, reset = safety_request_for(run_mode, tick)
        node.send_output(
            "safety_request",
            pa.array([int(emergency_stop), int(reset)], type=pa.uint8()),
            metadata={"operator_tick": tick, "run_mode": run_mode},
        )
        tick += 1


if __name__ == "__main__":
    main()
