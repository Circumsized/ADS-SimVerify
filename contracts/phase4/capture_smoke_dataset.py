#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE3_MANIFEST = ROOT / "contracts/phase3/domain_scene_baseline.json"
OVERLAY_MANIFEST = ROOT / "assets/phase4/warehouse_nav14_overlay.manifest.json"
OVERLAY_USD = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from warehouse_semantics import CHANNELS, FREE_CHANNELS, TAXONOMY_ID, channel_id, classify_prim_path  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_semantic_labels(value):
    if isinstance(value, dict):
        return [str(item) for item in value.values()]
    text = str(value)
    try:
        decoded = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return [text]
    return parse_semantic_labels(decoded)


def semantic_id_image(raw, np):
    ids = np.asarray(raw.get("data"))
    if ids.ndim != 2:
        raise RuntimeError(f"semantic annotator returned shape {ids.shape}")
    lookup = {}
    for raw_id, value in raw.get("info", {}).get("idToLabels", {}).items():
        labels = parse_semantic_labels(value)
        label_text = " ".join(labels)
        selected = next((label for label in CHANNELS if label in label_text), None)
        lookup[int(raw_id)] = {
            "class_id": channel_id(selected),
            "source_labels": labels,
        }
    result = np.full(ids.shape, channel_id("unknown_or_unlabeled"), dtype=np.uint8)
    for raw_id, mapping in lookup.items():
        result[ids == raw_id] = mapping["class_id"]
    return result, lookup


def build_ipm_remap(np, cv2, sensor, bev):
    height, width = sensor["image_size"][1], sensor["image_size"][0]
    fx, fy = sensor["intrinsics"]["fx"], sensor["intrinsics"]["fy"]
    cx, cy = sensor["intrinsics"]["cx"], sensor["intrinsics"]["cy"]
    ext = sensor["body_extrinsics"]
    rows, cols = bev["shape"][0], bev["shape"][1]
    meters = bev["meters_per_cell"]
    ego_row, ego_col = bev["ego_origin_cell"]
    map_x = np.full((rows, cols), -1.0, dtype=np.float32)
    map_y = np.full((rows, cols), -1.0, dtype=np.float32)
    sy, cyaw = math.sin(ext["yaw_rad"]), math.cos(ext["yaw_rad"])
    sp, cp = math.sin(ext["pitch_rad"]), math.cos(ext["pitch_rad"])
    sr, cr = math.sin(ext["roll_rad"]), math.cos(ext["roll_rad"])
    for row in range(rows):
        for col in range(cols):
            forward = (ego_row - row) * meters
            left = (ego_col - col) * meters
            if forward < 0.2:
                continue
            relative_forward = forward - ext["forward_m"]
            relative_left = left - ext["left_m"]
            heading_forward = cyaw * relative_forward + sy * relative_left
            heading_left = -sy * relative_forward + cyaw * relative_left
            if heading_forward <= 0.0:
                continue
            x_level = -heading_left
            y_pitched = ext["height_m"] * cp - heading_forward * sp
            z_camera = ext["height_m"] * sp + heading_forward * cp
            if z_camera <= 1e-6:
                continue
            x_camera = cr * x_level + sr * y_pitched
            y_camera = -sr * x_level + cr * y_pitched
            u = fx * x_camera / z_camera + cx
            v = fy * y_camera / z_camera + cy
            if 0.0 <= u < width and 0.0 <= v < height:
                map_x[row, col] = u
                map_y[row, col] = v
    return map_x, map_y


def one_hot(class_ids, np):
    return np.stack([class_ids == class_id for class_id in range(len(CHANNELS))]).astype(np.uint8)


def occupancy_metrics(candidate, oracle, valid, np):
    candidate_occupied = ~candidate[list(FREE_CHANNELS)].any(axis=0)
    oracle_occupied = ~oracle[list(FREE_CHANNELS)].any(axis=0)
    valid_count = max(int(valid.sum()), 1)
    occupied_union = int(((candidate_occupied | oracle_occupied) & valid).sum())
    free_union = int((((~candidate_occupied) | (~oracle_occupied)) & valid).sum())
    return {
        "ipm_roi_occupied_ratio": float(candidate_occupied[valid].mean()),
        "oracle_roi_occupied_ratio": float(oracle_occupied[valid].mean()),
        "occupied_iou": float(
            ((candidate_occupied & oracle_occupied) & valid).sum() / max(occupied_union, 1)
        ),
        "free_iou": float(
            (((~candidate_occupied) & (~oracle_occupied)) & valid).sum() / max(free_union, 1)
        ),
        "false_free_rate": float(((~candidate_occupied) & oracle_occupied & valid).sum() / valid_count),
        "false_occupied_rate": float((candidate_occupied & (~oracle_occupied) & valid).sum() / valid_count),
    }


def mean_metrics(records):
    return {
        name: sum(record[name] for record in records) / len(records)
        for name in records[0]
    }


def write_evidence(path, rgb, semantic_ids, depth, ipm, ipm_valid, depth_bev, depth_valid, oracle, np, cv2):
    palette = np.asarray(
        [
            (45, 170, 85), (210, 210, 65), (220, 130, 45), (155, 85, 75),
            (75, 120, 180), (215, 145, 40), (145, 110, 185), (225, 190, 60),
            (135, 90, 45), (210, 80, 90), (80, 170, 210), (70, 150, 170),
            (180, 80, 55), (70, 70, 70),
        ],
        dtype=np.uint8,
    )

    def labeled(image, title):
        panel = cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(panel, (0, 0), (319, 28), (15, 15, 15), -1)
        cv2.putText(panel, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    def bev_image(bev_tensor, valid):
        free = bev_tensor[list(FREE_CHANNELS)].any(axis=0)
        image = np.full((*free.shape, 3), (35, 35, 35), dtype=np.uint8)
        image[valid & free] = (45, 170, 85)
        image[valid & (~free)] = (215, 65, 65)
        return image

    finite = np.isfinite(depth)
    depth_vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if finite.any():
        limit = float(np.percentile(depth[finite], 95))
        normalized = np.clip(depth / max(limit, 1e-6), 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        depth_vis = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        depth_vis[~finite] = 0
    panels = (
        labeled(rgb, "RGB8"),
        labeled(palette[semantic_ids], "warehouse_nav14_v1"),
        labeled(depth_vis, "metric depth"),
        labeled(bev_image(ipm, ipm_valid), "flat IPM"),
        labeled(bev_image(depth_bev, depth_valid), "depth-lift BEV"),
        labeled(bev_image(oracle, np.ones(ipm_valid.shape, dtype=bool)), "USD oracle"),
    )
    canvas = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("evidence PNG encoding failed")
    path.write_bytes(encoded.tobytes())


def depth_lift_semantic_bev(semantic_ids, depth, sensor, bev, np):
    height, width = semantic_ids.shape
    intrinsics = sensor["intrinsics"]
    ext = sensor["body_extrinsics"]
    pixel_v, pixel_u = np.indices((height, width), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)
    safe_depth = np.where(valid, depth, 0.0)
    x_camera = (pixel_u - intrinsics["cx"]) * safe_depth / intrinsics["fx"]
    y_camera = (pixel_v - intrinsics["cy"]) * safe_depth / intrinsics["fy"]
    z_camera = safe_depth

    sr, cr = math.sin(ext["roll_rad"]), math.cos(ext["roll_rad"])
    sp, cp = math.sin(ext["pitch_rad"]), math.cos(ext["pitch_rad"])
    sy, cyaw = math.sin(ext["yaw_rad"]), math.cos(ext["yaw_rad"])
    x_level = cr * x_camera - sr * y_camera
    y_pitched = sr * x_camera + cr * y_camera
    heading_forward = -sp * y_pitched + cp * z_camera
    heading_left = -x_level
    relative_forward = cyaw * heading_forward - sy * heading_left
    relative_left = sy * heading_forward + cyaw * heading_left
    forward = relative_forward + ext["forward_m"]
    left = relative_left + ext["left_m"]

    rows, cols = bev["shape"][0], bev["shape"][1]
    cell_row = np.rint(bev["ego_origin_cell"][0] - forward / bev["meters_per_cell"]).astype(np.int32)
    cell_col = np.rint(bev["ego_origin_cell"][1] - left / bev["meters_per_cell"]).astype(np.int32)
    valid &= (cell_row >= 0) & (cell_row < rows) & (cell_col >= 0) & (cell_col < cols)

    result = np.full((rows, cols), channel_id("unknown_or_unlabeled"), dtype=np.uint8)
    observed = np.zeros((rows, cols), dtype=bool)
    # Unknown is lowest priority, then free space, then occupied classes.
    class_priority = [13, 0, 1, *range(2, 13)]
    for class_index in class_priority:
        selected = valid & (semantic_ids == class_index)
        result[cell_row[selected], cell_col[selected]] = class_index
        observed[cell_row[selected], cell_col[selected]] = True
    return one_hot(result, np), observed


def geometry_oracle(stage, pose, bev, np, cv2, Usd, UsdGeom):
    rows, cols = bev["shape"][0], bev["shape"][1]
    meters = bev["meters_per_cell"]
    ego_row, ego_col = bev["ego_origin_cell"]
    result = np.zeros((rows, cols), dtype=np.uint8)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    cos_yaw, sin_yaw = math.cos(pose[3]), math.sin(pose[3])
    entries = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        class_id = channel_id(classify_prim_path(str(prim.GetPath())))
        if class_id == 0:
            continue
        try:
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            low, high = box.GetMin(), box.GetMax()
        except Exception:
            continue
        if float(high[2]) < -0.02 or float(low[2]) > 0.80:
            continue
        cells = []
        for world_x, world_y in (
            (float(low[0]), float(low[1])),
            (float(low[0]), float(high[1])),
            (float(high[0]), float(high[1])),
            (float(high[0]), float(low[1])),
        ):
            dx, dy = world_x - pose[0], world_y - pose[1]
            forward = dx * cos_yaw + dy * sin_yaw
            left = -dx * sin_yaw + dy * cos_yaw
            row = int(round(ego_row - forward / meters))
            col = int(round(ego_col - left / meters))
            cells.append((col, row))
        entries.append((class_id, cells))
    for class_id, cells in sorted(entries, key=lambda item: item[0] not in FREE_CHANNELS):
        cv2.fillConvexPoly(result, np.asarray(cells, dtype=np.int32), int(class_id))
    return one_hot(result, np)


def camera_matrix(eye, yaw, pitch, Gf):
    distance = 5.0
    target = Gf.Vec3d(
        eye[0] + math.cos(yaw) * distance,
        eye[1] + math.sin(yaw) * distance,
        eye[2] - math.tan(pitch) * distance,
    )
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), target, Gf.Vec3d(0, 0, 1)).GetInverse()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be positive")
    if not OVERLAY_USD.is_file() or not OVERLAY_MANIFEST.is_file():
        raise SystemExit("run build_semantic_overlay.py first")

    phase3 = json.loads(PHASE3_MANIFEST.read_text(encoding="utf-8"))
    overlay = json.loads(OVERLAY_MANIFEST.read_text(encoding="utf-8"))
    output = args.output or ROOT / "artifacts/phase4_capture" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import cv2
        import numpy as np
        import omni
        import omni.replicator.core as rep
        from pxr import Gf, Usd, UsdGeom
        from isaacsim.core.utils.stage import open_stage

        open_stage(str(OVERLAY_USD))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
            raise RuntimeError("overlay stage metersPerUnit must be 1.0")
        if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
            raise RuntimeError("overlay stage upAxis must be Z")
        if not stage.GetDefaultPrim() or str(stage.GetDefaultPrim().GetPath()) != "/Root":
            raise RuntimeError("overlay stage defaultPrim must be /Root")
        camera = UsdGeom.Camera.Define(stage, "/Phase4CaptureCamera")
        sensor = phase3["sensor_geometry"]
        width, height = sensor["image_size"]
        horizontal_aperture = 20.955
        vertical_aperture = (
            sensor["intrinsics"]["fx"]
            * horizontal_aperture
            * height
            / (sensor["intrinsics"]["fy"] * width)
        )
        focal_length = sensor["intrinsics"]["fx"] * horizontal_aperture / width
        camera.CreateHorizontalApertureAttr(horizontal_aperture)
        camera.CreateVerticalApertureAttr(vertical_aperture)
        camera.CreateFocalLengthAttr(focal_length)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        transform = UsdGeom.Xformable(camera).AddTransformOp()
        render_product = rep.create.render_product(str(camera.GetPath()), (width, height))
        annotators = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "semantic": rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"colorize": False}
            ),
            "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
        }
        for annotator in annotators.values():
            annotator.attach([render_product])

        map_x, map_y = build_ipm_remap(np, cv2, sensor, phase3["bev_contract"])
        ipm_valid_mask = map_x >= 0.0
        poses = (
            (4.0, -1.0, 0.0),
            (4.0, -3.0, math.pi / 2),
            (4.0, 2.0, math.pi),
            (4.0, 5.0, 0.0),
            (4.0, 8.0, math.pi),
            (4.0, 10.5, math.pi / 2),
            (4.0, 12.0, -math.pi / 2),
            (4.0, 3.0, -math.pi / 2),
        )
        frames = []
        occupied_ratios = []
        roi_occupied_ratios = []
        depth_roi_occupied_ratios = []
        ipm_agreements = []
        depth_agreements = []
        evidence_arrays = None
        for index in range(args.frames):
            x, y, yaw = poses[index % len(poses)]
            ext = sensor["body_extrinsics"]
            eye = (
                x + math.cos(yaw) * ext["forward_m"] - math.sin(yaw) * ext["left_m"],
                y + math.sin(yaw) * ext["forward_m"] + math.cos(yaw) * ext["left_m"],
                ext["height_m"],
            )
            transform.Set(camera_matrix(eye, yaw + ext["yaw_rad"], ext["pitch_rad"], Gf))
            rep.orchestrator.step()
            rgb = np.asarray(annotators["rgb"].get_data())[:, :, :3].astype(np.uint8)
            semantic_ids, semantic_lookup = semantic_id_image(annotators["semantic"].get_data(), np)
            depth = np.asarray(annotators["depth"].get_data(), dtype=np.float32)
            ipm_ids = cv2.remap(
                semantic_ids,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=channel_id("unknown_or_unlabeled"),
            )
            ipm = one_hot(ipm_ids, np)
            oracle = geometry_oracle(
                stage,
                (x, y, eye[2], yaw),
                phase3["bev_contract"],
                np,
                cv2,
                Usd,
                UsdGeom,
            )
            agreement = occupancy_metrics(ipm, oracle, ipm_valid_mask, np)
            depth_bev, depth_observed_mask = depth_lift_semantic_bev(
                semantic_ids,
                depth,
                sensor,
                phase3["bev_contract"],
                np,
            )
            depth_agreement = occupancy_metrics(depth_bev, oracle, depth_observed_mask, np)
            ipm_agreements.append(agreement)
            depth_agreements.append(depth_agreement)
            frame_id = f"frame_{index:06d}"
            archive = output / f"{frame_id}.npz"
            np.savez_compressed(
                archive,
                rgb8=rgb,
                semantic_ids=semantic_ids,
                depth_to_image_plane=depth,
                semantic_gt_ipm=ipm,
                usd_geometry_oracle=oracle,
                ipm_valid_mask=ipm_valid_mask,
                semantic_depth_bev=depth_bev,
                depth_observed_mask=depth_observed_mask,
            )
            jpeg = output / f"{frame_id}.jpg"
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            jpeg.write_bytes(encoded.tobytes())
            occupied = 1.0 - float(ipm[list(FREE_CHANNELS)].any(axis=0).mean())
            class_pixel_counts = {
                name: int((semantic_ids == class_index).sum())
                for class_index, name in enumerate(CHANNELS)
            }
            occupied_ratios.append(occupied)
            roi_occupied_ratios.append(agreement["ipm_roi_occupied_ratio"])
            depth_roi_occupied_ratios.append(depth_agreement["ipm_roi_occupied_ratio"])
            finite_depth = depth[np.isfinite(depth)]
            frames.append(
                {
                    "source_frame_id": index,
                    "simulation_time_s": index / phase3["bev_contract"]["frequency_hz"],
                    "pose_world_xy_yaw": [x, y, yaw],
                    "archive": archive.name,
                    "archive_sha256": sha256(archive),
                    "jpeg": jpeg.name,
                    "jpeg_sha256": sha256(jpeg),
                    "semantic_raw_id_count": len(semantic_lookup),
                    "semantic_raw_id_map": semantic_lookup,
                    "semantic_class_pixel_counts": class_pixel_counts,
                    "ipm_occupied_ratio": occupied,
                    "oracle_agreement_valid_roi": agreement,
                    "depth_oracle_agreement_observed_roi": depth_agreement,
                    "depth_finite_ratio": float(np.isfinite(depth).mean()),
                    "depth_p50_m": float(np.median(finite_depth)),
                    "depth_p95_m": float(np.percentile(finite_depth, 95)),
                }
            )
            if index == 0:
                evidence_arrays = (
                    rgb,
                    semantic_ids,
                    depth,
                    ipm,
                    ipm_valid_mask,
                    depth_bev,
                    depth_observed_mask,
                    oracle,
                )

        evidence = output / "evidence.png"
        write_evidence(evidence, *evidence_arrays, np, cv2)

        summary = {
            "schema_version": "phase4-capture-v1",
            "status": "smoke_evidence_not_model_gate",
            "taxonomy_id": TAXONOMY_ID,
            "frame_count": len(frames),
            "frequency_hz": phase3["bev_contract"]["frequency_hz"],
            "shapes": {
                "rgb8": [height, width, 3],
                "semantic_ids": [height, width],
                "depth_to_image_plane": [height, width],
                "semantic_gt_ipm": [14, 192, 192],
                "usd_geometry_oracle": [14, 192, 192],
                "ipm_valid_mask": [192, 192],
                "semantic_depth_bev": [14, 192, 192],
                "depth_observed_mask": [192, 192],
            },
            "array_color_order": "RGB8",
            "jpeg_decoded_color_order": "BGR8",
            "usd_camera": {
                "horizontal_aperture_mm": horizontal_aperture,
                "vertical_aperture_mm": vertical_aperture,
                "focal_length_mm": focal_length,
                "effective_fx": focal_length * width / horizontal_aperture,
                "effective_fy": focal_length * height / vertical_aperture,
            },
            "camera": sensor,
            "overlay_manifest_sha256": sha256(OVERLAY_MANIFEST),
            "overlay_usd_sha256": sha256(OVERLAY_USD),
            "ipm_occupied_ratio": {
                "mean": sum(occupied_ratios) / len(occupied_ratios),
                "min": min(occupied_ratios),
                "max": max(occupied_ratios),
            },
            "ipm_valid_ratio": float(ipm_valid_mask.mean()),
            "ipm_roi_occupied_ratio": {
                "mean": sum(roi_occupied_ratios) / len(roi_occupied_ratios),
                "min": min(roi_occupied_ratios),
                "max": max(roi_occupied_ratios),
            },
            "depth_roi_occupied_ratio": {
                "mean": sum(depth_roi_occupied_ratios) / len(depth_roi_occupied_ratios),
                "min": min(depth_roi_occupied_ratios),
                "max": max(depth_roi_occupied_ratios),
            },
            "oracle_agreement_mean": {
                "semantic_gt_ipm": mean_metrics(ipm_agreements),
                "semantic_depth_bev": mean_metrics(depth_agreements),
            },
            "evidence": evidence.name,
            "evidence_sha256": sha256(evidence),
            "candidate_perception": None,
            "frames": frames,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        print(json.dumps(summary, indent=2))
        print(f"Phase 4 smoke capture written: {output}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
