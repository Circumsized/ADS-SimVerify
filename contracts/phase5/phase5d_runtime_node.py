#!/usr/bin/env python3
import json
import os
import sys
import time
from collections import OrderedDict
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


def metadata_value(event, key):
    metadata = event.get("metadata") or {}
    if key in metadata:
        return metadata[key]
    return (metadata.get("parameters") or {}).get(key)


def source_frame_id(event):
    try:
        return int(metadata_value(event, "source_frame_id"))
    except (TypeError, ValueError):
        return None


class FramePairer:
    def __init__(self, max_pending):
        self.max_pending = max_pending
        self.pending = {"jpeg_image": OrderedDict(), "metric_depth": OrderedDict()}

    def put(self, stream, frame_id, value):
        if stream not in self.pending or frame_id is None:
            return None, []
        self.pending[stream][frame_id] = value
        other = "metric_depth" if stream == "jpeg_image" else "jpeg_image"
        if frame_id in self.pending[other]:
            jpeg = self.pending["jpeg_image"].pop(frame_id)
            depth = self.pending["metric_depth"].pop(frame_id)
            return (frame_id, jpeg, depth), []
        evicted = []
        while len(self.pending[stream]) > self.max_pending:
            evicted.append(self.pending[stream].popitem(last=False)[0])
        return None, evicted


def output_metadata(frame_id, valid, latency_ms, reason=None):
    metadata = {
        "source_frame_id": frame_id,
        "source_kind": "warehouse_nav14_depth_lift_shadow",
        "shape": [192, 192],
        "semantic_taxonomy": "warehouse_nav14_v1",
        "valid": valid,
        "latency_ms": latency_ms,
        "candidate_controls_vehicle": False,
    }
    if reason:
        metadata["error"] = reason
    return metadata


def main():
    contract = json.loads((ROOT / "contracts/phase5/phase5d_contract.json").read_text())
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    manifest_path = ROOT / "model/warehouse_nav14_candidate.json"
    manifest = json.loads(manifest_path.read_text())
    model_path = ROOT / manifest["model"]
    if sha256(model_path) != manifest["model_sha256"]:
        raise SystemExit("Phase 5-D candidate hash mismatch")
    model = WarehouseCandidate(model_path)
    warmup_image = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.infer(warmup_image)
    pairer = FramePairer(contract["runtime"]["max_pending_frames"])
    sensor, bev = phase3["sensor_geometry"], phase3["bev_contract"]
    expected_depth_size = sensor["image_size"][0] * sensor["image_size"][1]
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
        frame_id = source_frame_id(event)
        value = event["value"].to_numpy().copy()
        pair, evicted = pairer.put(stream, frame_id, value)
        for stale_id in evicted:
            occupied = np.full((192, 192), 255, dtype=np.uint8)
            metadata = output_metadata(stale_id, False, 0.0, "unpaired input evicted")
            node.send_output("shadow_bev_grid", pa.array(occupied.ravel()), metadata=metadata)
            node.send_output("shadow_health", pa.array([stale_id, 0.0, 0.0], type=pa.float32()), metadata=metadata)
        if pair is None:
            continue
        frame_id, jpeg_data, depth_data = pair
        started = time.perf_counter_ns()
        try:
            decoded = cv2.imdecode(jpeg_data.astype(np.uint8, copy=False), cv2.IMREAD_COLOR)
            if decoded is None or decoded.shape != (480, 640, 3):
                raise ValueError(f"JPEG shape mismatch: {None if decoded is None else decoded.shape}")
            if depth_data.size != expected_depth_size:
                raise ValueError(f"depth size mismatch: {depth_data.size}")
            depth = depth_data.astype(np.float32, copy=False).reshape(480, 640)
            classes, _ = model.infer(decoded)
            semantic, _ = candidate_depth_lift(classes, depth, sensor, bev)
            occupied = np.where(occupancy_from_semantic(semantic), 255, 0).astype(np.uint8)
            valid, reason = True, None
        except Exception as error:
            occupied = np.full((192, 192), 255, dtype=np.uint8)
            valid, reason = False, str(error)
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        metadata = output_metadata(frame_id, valid, latency_ms, reason)
        node.send_output("shadow_bev_grid", pa.array(occupied.ravel()), metadata=metadata)
        node.send_output("shadow_health", pa.array([frame_id, float(valid), latency_ms], type=pa.float32()), metadata=metadata)
        processed += 1
        if processed % 20 == 0:
            print(f"[Phase 5-D runtime] frames={processed} latency={latency_ms:.2f}ms valid={int(valid)}")


if __name__ == "__main__":
    main()
