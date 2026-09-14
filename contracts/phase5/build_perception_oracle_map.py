#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE5A_STATUS = ROOT / "contracts/phase5/phase5a_status.json"
OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
PHASE4_DIR = ROOT / "contracts/phase4"
sys.path.insert(0, str(PHASE4_DIR))
from warehouse_semantics import FREE_CHANNELS, channel_id, classify_prim_path  # noqa: E402

MIN_HEIGHT_M = 0.02
MAX_HEIGHT_M = 0.35


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_polygon_z(polygon, height, keep_above):
    clipped = []
    if not polygon:
        return clipped
    for start, end in zip(polygon, polygon[1:] + polygon[:1]):
        start_inside = start[2] >= height if keep_above else start[2] <= height
        end_inside = end[2] >= height if keep_above else end[2] <= height
        if start_inside:
            clipped.append(start)
        if start_inside != end_inside:
            ratio = (height - start[2]) / (end[2] - start[2])
            clipped.append(
                (
                    start[0] + ratio * (end[0] - start[0]),
                    start[1] + ratio * (end[1] - start[1]),
                    height,
                )
            )
    return clipped


def clip_polygon_height_band(polygon, minimum, maximum):
    return clip_polygon_z(
        clip_polygon_z(polygon, minimum, keep_above=True),
        maximum,
        keep_above=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts/phase5c2_oracle" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)

    phase5a = json.loads(PHASE5A_STATUS.read_text(encoding="utf-8"))
    control_manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    control_manifest = json.loads(control_manifest_path.read_text(encoding="utf-8"))
    bounds = control_manifest["bounds_xy_m"]
    resolution = control_manifest["resolution_m"]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import cv2
        import numpy as np
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(OVERLAY), load=Usd.Stage.LoadAll)
        min_x, max_x, min_y, max_y = bounds
        rows = int(round((max_y - min_y) / resolution)) + 1
        cols = int(round((max_x - min_x) / resolution)) + 1
        occupied = np.zeros((rows, cols), dtype=np.uint8)
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        class_counts = Counter()
        face_count = 0
        clipped_face_count = 0
        contributing_meshes = 0

        def grid_point(point):
            col = int(round((point[0] - min_x) / resolution))
            row = int(round((max_y - point[1]) / resolution))
            return col, row

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            label = classify_prim_path(str(prim.GetPath()))
            class_counts[label] += 1
            if channel_id(label) in FREE_CHANNELS:
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices:
                continue
            transform = xform_cache.GetLocalToWorldTransform(prim)
            world_points = [
                transform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                for p in points
            ]
            offset = 0
            contributed = False
            for count in counts:
                polygon = [tuple(world_points[int(index)]) for index in indices[offset : offset + count]]
                offset += count
                face_count += 1
                polygon = clip_polygon_height_band(
                    polygon, MIN_HEIGHT_M, MAX_HEIGHT_M
                )
                if len(polygon) < 2:
                    continue
                pixels = np.asarray([grid_point(point) for point in polygon], dtype=np.int32)
                cv2.fillConvexPoly(occupied, pixels, 1)
                cv2.polylines(occupied, [pixels], True, 1, 1)
                clipped_face_count += 1
                contributed = True
            contributing_meshes += int(contributed)

        archive = output / "perception_oracle_map.npz"
        np.savez_compressed(archive, perception_occupied=occupied)
        preview = np.full((*occupied.shape, 3), (245, 245, 245), dtype=np.uint8)
        preview[occupied > 0] = (35, 35, 35)
        preview_path = output / "perception_oracle_map.png"
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError("failed to write perception Oracle preview")
        manifest = {
            "schema_version": "phase5c2-perception-oracle-v1",
            "status": "perception_scoring_only",
            "control_authority": False,
            "source_overlay": str(OVERLAY.relative_to(ROOT)),
            "source_overlay_sha256": sha256(OVERLAY),
            "phase5a_control_manifest": str(control_manifest_path.relative_to(ROOT)),
            "phase5a_control_manifest_sha256": sha256(control_manifest_path),
            "geometry": "exact USD mesh faces clipped to robot collision-height band",
            "height_band_m": [MIN_HEIGHT_M, MAX_HEIGHT_M],
            "bounds_xy_m": bounds,
            "resolution_m": resolution,
            "shape": [rows, cols],
            "contributing_meshes": contributing_meshes,
            "source_face_count": face_count,
            "clipped_face_count": clipped_face_count,
            "class_mesh_counts": dict(sorted(class_counts.items())),
            "occupied_ratio": float(occupied.mean()),
            "archive": archive.name,
            "archive_sha256": sha256(archive),
            "preview": preview_path.name,
            "preview_sha256": sha256(preview_path),
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        print(json.dumps(manifest, indent=2))
        print(f"Phase 5-C2 perception Oracle: {output}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
