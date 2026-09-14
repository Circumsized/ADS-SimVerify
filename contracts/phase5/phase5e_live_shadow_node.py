#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contracts/phase5"))
from phase5b_shadow_replay import occupancy_from_semantic  # noqa: E402
from phase5c3_candidate_shadow import WarehouseCandidate, candidate_depth_lift  # noqa: E402
from phase5c_dynamic_upper_bound import sha256  # noqa: E402
from phase5d_runtime_node import FramePairer, output_metadata, source_frame_id  # noqa: E402


def publish(node, frame_id, occupied, observed, valid, latency_ms, reason=None):
    metadata = output_metadata(frame_id, valid, latency_ms, reason)
    metadata["source_kind"] = "phase5e_live_warehouse_shadow"
    node.send_output("shadow_bev_grid", pa.array(occupied.ravel()), metadata=metadata)
    node.send_output("shadow_valid", pa.array(observed.astype(np.uint8).ravel()), metadata=metadata)
    node.send_output(
        "shadow_health",
        pa.array([frame_id, float(valid), latency_ms], type=pa.float32()),
        metadata=metadata,
    )


def main():
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    manifest = json.loads((ROOT / "model/warehouse_nav14_candidate.json").read_text())
    model_path = ROOT / manifest["model"]
    if sha256(model_path) != manifest["model_sha256"]:
        raise SystemExit("Phase 5-E candidate hash mismatch")
    model = WarehouseCandidate(model_path)
    warmup = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.infer(warmup)
    pairer = FramePairer(4)
    sensor, bev = phase3["sensor_geometry"], phase3["bev_contract"]
    node = Node()
    processed = 0
    while True:
        event = node.next(timeout=0.1)
        if event is None:
            continue
        if event["type"] == "STOP":
            break
        if event["type"] != "INPUT" or event["id"] not in pairer.pending:
            continue
        stream = event["id"]
        pair, evicted = pairer.put(
            stream, source_frame_id(event), event["value"].to_numpy().copy()
        )
        for stale_id in evicted:
            publish(
                node,
                stale_id,
                np.full((192, 192), 255, dtype=np.uint8),
                np.zeros((192, 192), dtype=bool),
                False,
                0.0,
                "unpaired live input evicted",
            )
        if pair is None:
            continue
        frame_id, jpeg, depth_flat = pair
        started = time.perf_counter_ns()
        try:
            decoded = cv2.imdecode(jpeg.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (480, 640, 3):
                raise ValueError("live JPEG shape mismatch")
            if depth_flat.size != 480 * 640:
                raise ValueError("live metric depth shape mismatch")
            depth = depth_flat.astype(np.float32, copy=False).reshape(480, 640)
            classes, _ = model.infer(decoded)
            semantic, observed = candidate_depth_lift(classes, depth, sensor, bev)
            occupied = np.where(occupancy_from_semantic(semantic), 255, 0).astype(np.uint8)
            valid, reason = True, None
        except Exception as error:
            occupied = np.full((192, 192), 255, dtype=np.uint8)
            observed = np.zeros((192, 192), dtype=bool)
            valid, reason = False, str(error)
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        publish(node, frame_id, occupied, observed, valid, latency_ms, reason)
        processed += 1
        if processed % 100 == 0:
            print(f"[Phase 5-E live shadow] frames={processed} latency={latency_ms:.2f}ms valid={int(valid)}")


if __name__ == "__main__":
    main()
