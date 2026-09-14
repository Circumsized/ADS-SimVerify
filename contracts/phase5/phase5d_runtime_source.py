#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from dora import Node


def main():
    fixture = Path(os.environ["PHASE5D_FIXTURE"]).resolve()
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    node = Node()
    sent = 0
    while sent < len(manifest["frames"]):
        event = node.next(timeout=1.0)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] != "tick":
            continue
        frame = manifest["frames"][sent]
        frame_id = int(frame["source_frame_id"])
        metadata = {
            "source_frame_id": frame_id,
            "source_kind": "phase5d_runtime_fixture",
            "image_shape": manifest["image_shape"],
            "depth_shape": manifest["depth_shape"],
        }
        jpeg = (fixture / frame["jpeg"]).read_bytes()
        depth = np.load(fixture / frame["depth"]).astype(np.float32, copy=False)
        node.send_output("jpeg_image", pa.array(np.frombuffer(jpeg, dtype=np.uint8)), metadata=metadata)
        node.send_output("metric_depth", pa.array(depth.ravel(), type=pa.float32()), metadata=metadata)
        sent += 1
    print(f"[Phase 5-D source] emitted {sent} synchronized fixture frames")


if __name__ == "__main__":
    main()
