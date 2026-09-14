#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE4_STATUS = ROOT / "contracts/phase4/phase4_status.json"
OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
sys.path.insert(0, str(ROOT / "contracts/phase4"))
from warehouse_semantics import FREE_CHANNELS, channel_id, classify_prim_path  # noqa: E402

BOUNDS = (-12.0, 12.0, -6.0, 16.0)
RESOLUTION_M = 0.10
ROBOT_HALF_LENGTH_M = 0.18
ROBOT_HALF_WIDTH_M = 0.13
SAFETY_MARGIN_M = 0.08
ROBOT_HEIGHT_M = 0.35


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_to_grid(x, y, shape):
    min_x, max_x, min_y, max_y = BOUNDS
    col = int(round((x - min_x) / RESOLUTION_M))
    row = int(round((max_y - y) / RESOLUTION_M))
    return max(0, min(shape[0] - 1, row)), max(0, min(shape[1] - 1, col))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts/phase5a_oracle" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    phase4 = json.loads(PHASE4_STATUS.read_text(encoding="utf-8"))
    if phase4["p4_a_semantic_overlay"]["overlay_sha256"] != sha256(OVERLAY):
        raise SystemExit("Phase 4 semantic overlay hash mismatch")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import cv2
        import numpy as np
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(OVERLAY), load=Usd.Stage.LoadAll)
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError("oracle map requires a Z-up meter-scale stage")

        min_x, max_x, min_y, max_y = BOUNDS
        width = int(round((max_x - min_x) / RESOLUTION_M)) + 1
        height = int(round((max_y - min_y) / RESOLUTION_M)) + 1
        raw_occupied = np.zeros((height, width), dtype=np.uint8)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        class_counts = Counter()
        rasterized = 0

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            label = classify_prim_path(str(prim.GetPath()))
            class_index = channel_id(label)
            class_counts[label] += 1
            if class_index in FREE_CHANNELS:
                continue
            try:
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                low, high = box.GetMin(), box.GetMax()
            except Exception:
                continue
            if float(high[2]) < 0.02 or float(low[2]) > ROBOT_HEIGHT_M:
                continue
            x0, x1 = max(min_x, float(low[0])), min(max_x, float(high[0]))
            y0, y1 = max(min_y, float(low[1])), min(max_y, float(high[1]))
            if x0 >= x1 or y0 >= y1:
                continue
            row0, col0 = world_to_grid(x0, y1, raw_occupied.shape)
            row1, col1 = world_to_grid(x1, y0, raw_occupied.shape)
            cv2.rectangle(raw_occupied, (col0, row0), (col1, row1), 1, thickness=-1)
            rasterized += 1

        free_mask = (raw_occupied == 0).astype(np.uint8)
        clearance_m = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5) * RESOLUTION_M
        circumscribed_radius = (
            (ROBOT_HALF_LENGTH_M**2 + ROBOT_HALF_WIDTH_M**2) ** 0.5 + SAFETY_MARGIN_M
        )
        inflated_occupied = (clearance_m <= circumscribed_radius).astype(np.uint8)
        archive = output / "oracle_map.npz"
        np.savez_compressed(
            archive,
            raw_occupied=raw_occupied,
            inflated_occupied=inflated_occupied,
            clearance_m=clearance_m.astype(np.float32),
        )

        preview = np.full((*raw_occupied.shape, 3), (245, 245, 245), dtype=np.uint8)
        preview[inflated_occupied > 0] = (185, 185, 185)
        preview[raw_occupied > 0] = (35, 35, 35)
        preview_path = output / "oracle_map.png"
        ok, encoded = cv2.imencode(".png", preview)
        if not ok:
            raise RuntimeError("oracle map preview encoding failed")
        preview_path.write_bytes(encoded.tobytes())

        manifest = {
            "schema_version": "phase5a-oracle-map-v1",
            "source_overlay": str(OVERLAY.relative_to(ROOT)),
            "source_overlay_sha256": sha256(OVERLAY),
            "coordinate_frame": "W: x-forward-map-axis, y-left-map-axis, z-up",
            "bounds_xy_m": [min_x, max_x, min_y, max_y],
            "resolution_m": RESOLUTION_M,
            "shape": [height, width],
            "robot_footprint": {
                "half_length_m": ROBOT_HALF_LENGTH_M,
                "half_width_m": ROBOT_HALF_WIDTH_M,
                "safety_margin_m": SAFETY_MARGIN_M,
                "circumscribed_planning_radius_m": circumscribed_radius,
                "collision_check": "yaw-aware rectangle sampled against raw occupancy",
            },
            "rasterized_ground_intersecting_meshes": rasterized,
            "class_mesh_counts": dict(sorted(class_counts.items())),
            "raw_occupied_ratio": float(raw_occupied.mean()),
            "inflated_occupied_ratio": float(inflated_occupied.mean()),
            "archive": archive.name,
            "archive_sha256": sha256(archive),
            "preview": preview_path.name,
            "preview_sha256": sha256(preview_path),
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
        print(json.dumps(manifest, indent=2))
        print(f"Phase 5-A oracle map written: {output}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
