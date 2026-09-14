# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dora-rs==0.3.13",
#     "numpy>=1.26.0",
#     "opencv-python-headless>=4.8.0",
#     "pyarrow>=14.0.0"
# ]
# ///
import argparse
import csv
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from dora import Node

BEV_WIDTH = 192
BEV_HEIGHT = 192
BEV_METERS_PER_CELL = 20.0 / BEV_WIDTH
BEV_EGO_ROW = (BEV_HEIGHT - 1) * 0.5
BEV_EGO_COL = (BEV_WIDTH - 1) * 0.5
RING_SIZE = 64

CSV_FIELDS = [
    "wall_time_s",
    "frame_a",
    "frame_b",
    "frame_c",
    "exact_a",
    "exact_b",
    "skew_a_frames",
    "skew_b_frames",
    "b_valid",
    "c_valid",
    "c_latency_ms",
    "roi_occupied_ratio_a",
    "roi_occupied_ratio_b",
    "roi_occupied_ratio_c",
    "ab_occupied_iou",
    "ab_free_iou",
    "ab_false_free_rate",
    "ab_false_occupied_rate",
    "ab_agreement",
    "bc_occupied_iou",
    "bc_free_iou",
    "bc_false_free_rate",
    "bc_false_occupied_rate",
    "bc_agreement",
    "ac_occupied_iou",
    "ac_free_iou",
    "ac_false_free_rate",
    "ac_false_occupied_rate",
    "ac_agreement",
    "nearest_a_m",
    "nearest_b_m",
    "nearest_c_m",
]


def metadata_value(event, key):
    metadata = event.get("metadata") or {}
    if key in metadata:
        return metadata[key]
    parameters = metadata.get("parameters") or {}
    return parameters.get(key)


def source_frame_id(event):
    value = metadata_value(event, "source_frame_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def put_ring(ring, frame_id, value):
    if frame_id is None:
        return
    ring[frame_id] = value
    ring.move_to_end(frame_id)
    while len(ring) > RING_SIZE:
        ring.popitem(last=False)


def select_frame(ring, frame_id):
    if frame_id in ring:
        return frame_id, ring[frame_id], True
    if not ring:
        return None, None, False
    nearest_id = min(ring, key=lambda candidate: abs(candidate - frame_id))
    return nearest_id, ring[nearest_id], False


def control_roi_mask():
    rows, cols = np.indices((BEV_HEIGHT, BEV_WIDTH))
    forward = (BEV_EGO_ROW - rows) * BEV_METERS_PER_CELL
    left = (BEV_EGO_COL - cols) * BEV_METERS_PER_CELL
    return (forward >= 0.20) & (forward <= 2.20) & (np.abs(left) <= 0.80)


def pair_metrics(reference, candidate, roi):
    ref_occupied = (reference > 0) & roi
    candidate_occupied = (candidate > 0) & roi
    ref_free = (~ref_occupied) & roi
    candidate_free = (~candidate_occupied) & roi

    occupied_union = np.count_nonzero(ref_occupied | candidate_occupied)
    free_union = np.count_nonzero(ref_free | candidate_free)
    ref_occupied_count = np.count_nonzero(ref_occupied)
    ref_free_count = np.count_nonzero(ref_free)
    return {
        "occupied_iou": (
            np.count_nonzero(ref_occupied & candidate_occupied) / occupied_union
            if occupied_union
            else 1.0
        ),
        "free_iou": (
            np.count_nonzero(ref_free & candidate_free) / free_union if free_union else 1.0
        ),
        "false_free_rate": (
            np.count_nonzero(ref_occupied & candidate_free) / ref_occupied_count
            if ref_occupied_count
            else 0.0
        ),
        "false_occupied_rate": (
            np.count_nonzero(ref_free & candidate_occupied) / ref_free_count
            if ref_free_count
            else 0.0
        ),
        "agreement": np.count_nonzero((ref_occupied == candidate_occupied) & roi)
        / np.count_nonzero(roi),
    }


def occupied_ratio(bev, roi):
    return float(np.mean(bev[roi] > 0))


def nearest_center_obstacle(bev):
    rows, cols = np.indices((BEV_HEIGHT, BEV_WIDTH))
    forward = (BEV_EGO_ROW - rows) * BEV_METERS_PER_CELL
    left = (BEV_EGO_COL - cols) * BEV_METERS_PER_CELL
    candidates = forward[
        (bev > 0)
        & (forward >= 0.20)
        & (forward <= 2.20)
        & (np.abs(left) <= 0.34)
    ]
    return float(candidates.min()) if candidates.size else None


def prefixed(prefix, metrics):
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def evidence_canvas(rgb_jpeg, grids):
    panels = []
    if rgb_jpeg is not None:
        rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is not None:
            panels.append(cv2.resize(rgb, (384, 288), interpolation=cv2.INTER_AREA))
    for label, grid in grids:
        panel = cv2.resize(grid, (288, 288), interpolation=cv2.INTER_NEAREST)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(panel)
    if not panels:
        return None
    target_height = min(panel.shape[0] for panel in panels)
    panels = [
        cv2.resize(panel, (round(panel.shape[1] * target_height / panel.shape[0]), target_height))
        for panel in panels
    ]
    return cv2.hconcat(panels)


class ProbeRecorder:
    def __init__(self, output_dir):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_dir) / stamp
        self.evidence_dir = self.run_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "frames.csv"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.rows = []
        self.evidence_count = 0

    def write(self, row, rgb_jpeg, grids):
        self.writer.writerow(row)
        self.csv_file.flush()
        self.rows.append(row)
        should_save = (
            self.evidence_count == 0
            or row["bc_false_free_rate"] >= 0.05
            or row["bc_false_occupied_rate"] >= 0.95
        )
        if should_save and self.evidence_count < 40:
            canvas = evidence_canvas(rgb_jpeg, grids)
            if canvas is not None:
                stem = self.evidence_dir / f"frame_{int(row['frame_c']):08d}"
                cv2.imwrite(
                    str(stem.with_suffix(".jpg")),
                    canvas,
                )
                for label, grid in grids:
                    suffix = label.lower().replace(" ", "_").replace("-", "_")
                    cv2.imwrite(str(stem.with_name(f"{stem.name}_{suffix}.png")), grid)
                self.evidence_count += 1

    def close(self):
        if self.csv_file.closed:
            return
        self.csv_file.close()
        summary = {
            "schema_version": "phase2-abc-probe-v1",
            "frames": len(self.rows),
            "exact_a_ratio": float(np.mean([row["exact_a"] for row in self.rows])) if self.rows else 0.0,
            "exact_b_ratio": float(np.mean([row["exact_b"] for row in self.rows])) if self.rows else 0.0,
            "b_valid_ratio": float(np.mean([row["b_valid"] for row in self.rows])) if self.rows else 0.0,
            "c_valid_ratio": float(np.mean([row["c_valid"] for row in self.rows])) if self.rows else 0.0,
            "metrics": {},
        }
        for field in CSV_FIELDS:
            if (
                field.endswith(("_iou", "_rate", "_agreement"))
                or "occupied_ratio" in field
                or field == "c_latency_ms"
            ):
                values = [float(row[field]) for row in self.rows]
                summary["metrics"][field] = {
                    "mean": float(np.mean(values)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                }
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Phase2 ABC Probe] artifacts: {self.run_dir}")


def self_test():
    roi = control_roi_mask()
    reference = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
    reference[82:86, 94:98] = 255
    identical = pair_metrics(reference, reference, roi)
    assert identical["agreement"] == 1.0
    assert identical["false_free_rate"] == 0.0
    missed = np.zeros_like(reference)
    metrics = pair_metrics(reference, missed, roi)
    assert metrics["false_free_rate"] == 1.0
    ring = OrderedDict([(10, "ten"), (12, "twelve")])
    assert select_frame(ring, 10) == (10, "ten", True)
    assert select_frame(ring, 11)[0] in (10, 12)
    assert source_frame_id({"metadata": {"source_frame_id": 42}}) == 42
    assert source_frame_id({"metadata": {"parameters": {"source_frame_id": 43}}}) == 43
    print("Phase 2 A/B/C probe self-test OK")


def main():
    parser = argparse.ArgumentParser(description="Passive same-frame A/B/C perception probe")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    output_dir = os.environ.get("PHASE2_PROBE_OUTPUT", "artifacts/phase2_probe")
    recorder = ProbeRecorder(output_dir)
    print("[Phase2 ABC Probe] passive only; no control output is declared.")
    node = Node()
    rings = {
        "a": OrderedDict(),
        "b": OrderedDict(),
        "b_valid": OrderedDict(),
        "rgb": OrderedDict(),
    }
    roi = control_roi_mask()
    missing_frame_ids = 0
    try:
        while True:
            event = node.next(timeout=0.05)
            if event is None:
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT":
                continue
            event_id = event["id"]
            frame_id = source_frame_id(event)
            data = event["value"].to_numpy()
            if event_id == "oracle_bev" and data.size == BEV_WIDTH * BEV_HEIGHT:
                put_ring(rings["a"], frame_id, data.reshape(BEV_HEIGHT, BEV_WIDTH).copy())
            elif event_id == "semantic_gt_bev" and data.size == BEV_WIDTH * BEV_HEIGHT:
                put_ring(rings["b"], frame_id, data.reshape(BEV_HEIGHT, BEV_WIDTH).copy())
            elif event_id == "semantic_gt_valid" and data.size:
                put_ring(rings["b_valid"], frame_id, bool(data[0]))
            elif event_id == "jpeg_image":
                put_ring(rings["rgb"], frame_id, data.astype(np.uint8, copy=False).tobytes())
            elif event_id != "pidnet_bev" or data.size != BEV_WIDTH * BEV_HEIGHT:
                continue

            if event_id != "pidnet_bev" or frame_id is None:
                if event_id == "pidnet_bev" and frame_id is None:
                    missing_frame_ids += 1
                    if missing_frame_ids % 20 == 1:
                        print("[Phase2 ABC Probe] C is missing source_frame_id; mismatched-frame comparison rejected.")
                continue
            frame_a, grid_a, exact_a = select_frame(rings["a"], frame_id)
            frame_b, grid_b, exact_b = select_frame(rings["b"], frame_id)
            _, b_valid, _ = select_frame(rings["b_valid"], frame_id)
            _, rgb_jpeg, _ = select_frame(rings["rgb"], frame_id)
            if grid_a is None or grid_b is None:
                continue
            grid_c = data.reshape(BEV_HEIGHT, BEV_WIDTH).copy()
            row = {
                "wall_time_s": time.time(),
                "frame_a": frame_a,
                "frame_b": frame_b,
                "frame_c": frame_id,
                "exact_a": int(exact_a),
                "exact_b": int(exact_b),
                "skew_a_frames": int(frame_a - frame_id),
                "skew_b_frames": int(frame_b - frame_id),
                "b_valid": int(bool(b_valid)),
                "c_valid": int(bool(metadata_value(event, "valid"))),
                "c_latency_ms": float(metadata_value(event, "latency_ms") or 0.0),
                "roi_occupied_ratio_a": occupied_ratio(grid_a, roi),
                "roi_occupied_ratio_b": occupied_ratio(grid_b, roi),
                "roi_occupied_ratio_c": occupied_ratio(grid_c, roi),
                **prefixed("ab", pair_metrics(grid_a, grid_b, roi)),
                **prefixed("bc", pair_metrics(grid_b, grid_c, roi)),
                **prefixed("ac", pair_metrics(grid_a, grid_c, roi)),
                "nearest_a_m": nearest_center_obstacle(grid_a),
                "nearest_b_m": nearest_center_obstacle(grid_b),
                "nearest_c_m": nearest_center_obstacle(grid_c),
            }
            recorder.write(row, rgb_jpeg, [("A USD", grid_a), ("B GT-IPM", grid_b), ("C PIDNet-IPM", grid_c)])
            if len(recorder.rows) % 20 == 0:
                print(
                    "[Phase2 ABC Probe] "
                    f"frames={len(recorder.rows)} exact=({row['exact_a']},{row['exact_b']}) "
                    f"B_valid={row['b_valid']} BC_false_free={row['bc_false_free_rate']:.3f} "
                    f"BC_IoU={row['bc_occupied_iou']:.3f}"
                )
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
