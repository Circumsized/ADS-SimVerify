#!/usr/bin/env python3
import os

import pyarrow as pa
from dora import Node


def main():
    node = Node()
    emergency_stop_once = os.environ.get("FSD_E_STOP_ONCE", "0") == "1"
    reset_once = os.environ.get("FSD_RESET_ONCE", "0") == "1"
    fired = False
    while True:
        event = node.next(timeout=1.0)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "tick":
            continue
        if not fired:
            node.send_output(
                "safety_request",
                pa.array([int(emergency_stop_once), int(reset_once)], type=pa.uint8()),
            )
            fired = True
        else:
            node.send_output("safety_request", pa.array([0, 0], type=pa.uint8()))


if __name__ == "__main__":
    main()