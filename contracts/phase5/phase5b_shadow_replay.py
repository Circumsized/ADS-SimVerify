#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
PHASE5A_STATUS = ROOT / "contracts/phase5/phase5a_status.json"
PIDNET_MODEL = ROOT / "model/pidnet_s.onnx"
PHASE4_DIR = ROOT / "contracts/phase4"
sys.path.insert(0, str(PHASE4_DIR))
from capture_smoke_dataset import (  # noqa: E402
    build_ipm_remap,
    camera_matrix,
    depth_lift_semantic_bev,
    semantic_id_image,
)
from warehouse_semantics import FREE_CHANNELS  # noqa: E402

METHODS = ("depth_gt", "pidnet_flat", "pidnet_depth")
CSV_FIELDS = [
    "source_frame_id",
    "scenario",
    "scenario_step",
    "simulation_time_s",
    "x_m",
    "y_m",
    "yaw_rad",
    "oracle_acceleration_mps2",
    "oracle_omega_radps",
    "oracle_velocity_mps",
    "common_roi_coverage",
    "depth_latency_ms",
    "candidate_latency_ms",
    "candidate_valid",
    "nearest_oracle_m",
    "nearest_depth_gt_m",
    "nearest_pidnet_flat_m",
    "nearest_pidnet_depth_m",
    "oracle_stop",
    "depth_gt_stop",
    "pidnet_flat_stop",
    "pidnet_depth_stop",
]
for method in METHODS:
    CSV_FIELDS.extend(
        [
            f"{method}_occupied_iou",
            f"{method}_free_iou",
            f"{method}_false_free_rate",
            f"{method}_false_occupied_rate",
            f"{method}_agreement",
            f"{method}_true_positive_count",
            f"{method}_true_negative_count",
            f"{method}_false_positive_count",
            f"{method}_false_negative_count",
        ]
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def occupancy_from_semantic(bev):
    return ~bev[list(FREE_CHANNELS)].any(axis=0)


def occupancy_metrics(oracle, candidate, valid):
    oracle = np.asarray(oracle, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    oracle_occupied = oracle & valid
    candidate_occupied = candidate & valid
    oracle_free = (~oracle) & valid
    candidate_free = (~candidate) & valid
    occupied_union = int(np.count_nonzero(oracle_occupied | candidate_occupied))
    free_union = int(np.count_nonzero(oracle_free | candidate_free))
    oracle_occupied_count = int(np.count_nonzero(oracle_occupied))
    oracle_free_count = int(np.count_nonzero(oracle_free))
    valid_count = int(np.count_nonzero(valid))
    true_positive = int(np.count_nonzero(oracle_occupied & candidate_occupied))
    true_negative = int(np.count_nonzero(oracle_free & candidate_free))
    false_positive = int(np.count_nonzero(oracle_free & candidate_occupied))
    false_negative = int(np.count_nonzero(oracle_occupied & candidate_free))
    return {
        "occupied_iou": (
            float(np.count_nonzero(oracle_occupied & candidate_occupied) / occupied_union)
            if occupied_union
            else 1.0
        ),
        "free_iou": (
            float(np.count_nonzero(oracle_free & candidate_free) / free_union)
            if free_union
            else 1.0
        ),
        "false_free_rate": (
            float(np.count_nonzero(oracle_occupied & candidate_free) / oracle_occupied_count)
            if oracle_occupied_count
            else 0.0
        ),
        "false_occupied_rate": (
            float(np.count_nonzero(oracle_free & candidate_occupied) / oracle_free_count)
            if oracle_free_count
            else 0.0
        ),
        "agreement": (
            float(np.count_nonzero((oracle == candidate) & valid) / valid_count)
            if valid_count
            else 0.0
        ),
        "true_positive_count": true_positive,
        "true_negative_count": true_negative,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
    }


def control_roi(bev):
    rows, cols = bev["shape"][0], bev["shape"][1]
    row, col = np.indices((rows, cols), dtype=np.float32)
    forward = (bev["ego_origin_cell"][0] - row) * bev["meters_per_cell"]
    left = (bev["ego_origin_cell"][1] - col) * bev["meters_per_cell"]
    return (forward >= 0.20) & (forward <= 2.20) & (np.abs(left) <= 0.80)


def nearest_center_obstacle(occupied, valid, bev):
    rows, cols = occupied.shape
    row, col = np.indices((rows, cols), dtype=np.float32)
    forward = (bev["ego_origin_cell"][0] - row) * bev["meters_per_cell"]
    left = (bev["ego_origin_cell"][1] - col) * bev["meters_per_cell"]
    selected = (
        occupied
        & valid
        & (forward >= 0.20)
        & (forward <= 2.20)
        & (np.abs(left) <= 0.34)
    )
    return float(forward[selected].min()) if np.any(selected) else None


def oracle_bev(raw_occupied, map_manifest, pose, bev):
    rows, cols = bev["shape"][0], bev["shape"][1]
    row, col = np.indices((rows, cols), dtype=np.float64)
    forward = (bev["ego_origin_cell"][0] - row) * bev["meters_per_cell"]
    left = (bev["ego_origin_cell"][1] - col) * bev["meters_per_cell"]
    x, y, yaw = pose
    world_x = x + forward * math.cos(yaw) - left * math.sin(yaw)
    world_y = y + forward * math.sin(yaw) + left * math.cos(yaw)
    min_x, max_x, min_y, max_y = map_manifest["bounds_xy_m"]
    resolution = map_manifest["resolution_m"]
    map_row = np.rint((max_y - world_y) / resolution).astype(np.int64)
    map_col = np.rint((world_x - min_x) / resolution).astype(np.int64)
    valid = (
        (map_row >= 0)
        & (map_row < raw_occupied.shape[0])
        & (map_col >= 0)
        & (map_col < raw_occupied.shape[1])
    )
    occupied = np.ones((rows, cols), dtype=bool)
    occupied[valid] = raw_occupied[map_row[valid], map_col[valid]] > 0
    return occupied, valid


def decode_pidnet(logits, confidence_threshold=0.50):
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 19:
        raise ValueError(f"PIDNet output must be NCHW with 19 classes, got {logits.shape}")
    scores = logits[0]
    best_class = np.argmax(scores, axis=0).astype(np.uint8)
    best_score = np.max(scores, axis=0, keepdims=True)
    confidence = 1.0 / np.exp(scores - best_score).sum(axis=0)
    return np.where(confidence >= confidence_threshold, best_class, 255).astype(np.uint8)


class CandidatePidnet:
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    @property
    def providers(self):
        return self.session.get_providers()

    def infer(self, decoded_bgr):
        started = time.perf_counter_ns()
        rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        logits = self.session.run(["output"], {"input": tensor})[0]
        class_map = decode_pidnet(logits)
        if class_map.shape != decoded_bgr.shape[:2]:
            class_map = cv2.resize(
                class_map,
                (decoded_bgr.shape[1], decoded_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        return class_map, latency_ms


def load_trajectory(phase5a_status):
    summary_path = ROOT / phase5a_status["closed_loop"]["summary"]["path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = []
    for scenario in summary["scenarios"]:
        telemetry_path = summary_path.parent / scenario["telemetry"]
        with telemetry_path.open(newline="", encoding="ascii") as source:
            for row in csv.DictReader(source):
                frames.append(
                    {
                        "scenario": scenario["name"],
                        "scenario_step": int(row["step"]),
                        "simulation_time_s": float(row["time_s"]),
                        "x_m": float(row["x_m"]),
                        "y_m": float(row["y_m"]),
                        "yaw_rad": float(row["yaw_rad"]),
                        "velocity_mps": float(row["velocity_mps"]),
                        "acceleration_mps2": float(row["acceleration_mps2"]),
                        "omega_radps": float(row["omega_radps"]),
                    }
                )
    return summary_path, frames


def aggregate(rows, field):
    values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def method_summary(rows, method, thresholds):
    metrics = {
        name: aggregate(rows, f"{method}_{name}")
        for name in (
            "occupied_iou",
            "free_iou",
            "false_free_rate",
            "false_occupied_rate",
            "agreement",
        )
    }
    dangerous_false_go = sum(
        int(row["oracle_stop"] and not row[f"{method}_stop"]) for row in rows
    )
    nuisance_false_stop = sum(
        int(not row["oracle_stop"] and row[f"{method}_stop"]) for row in rows
    )
    counts = {
        name: sum(int(row[f"{method}_{name}_count"]) for row in rows)
        for name in ("true_positive", "true_negative", "false_positive", "false_negative")
    }
    occupied_union = counts["true_positive"] + counts["false_positive"] + counts["false_negative"]
    free_union = counts["true_negative"] + counts["false_positive"] + counts["false_negative"]
    oracle_occupied = counts["true_positive"] + counts["false_negative"]
    oracle_free = counts["true_negative"] + counts["false_positive"]
    metrics["micro_confusion"] = {
        **counts,
        "occupied_iou": counts["true_positive"] / max(occupied_union, 1),
        "free_iou": counts["true_negative"] / max(free_union, 1),
        "false_free_rate": counts["false_negative"] / max(oracle_occupied, 1),
        "false_occupied_rate": counts["false_positive"] / max(oracle_free, 1),
    }
    metrics["dangerous_false_go_frames"] = dangerous_false_go
    metrics["nuisance_false_stop_frames"] = nuisance_false_stop
    metrics["gate_passed"] = bool(
        metrics["false_free_rate"]["mean"] <= thresholds["false_free_rate_mean_max"]
        and metrics["false_occupied_rate"]["mean"] <= thresholds["false_occupied_rate_mean_max"]
        and metrics["free_iou"]["mean"] >= thresholds["free_iou_mean_min"]
        and metrics["occupied_iou"]["mean"] >= thresholds["occupied_iou_mean_min"]
    )
    return metrics


def occupancy_panel(occupied, valid, label):
    image = np.full((*occupied.shape, 3), (32, 32, 32), dtype=np.uint8)
    image[valid & (~occupied)] = (45, 170, 85)
    image[valid & occupied] = (215, 65, 65)
    panel = cv2.resize(image, (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (319, 28), (15, 15, 15), -1)
    cv2.putText(panel, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def write_evidence(path, decoded_bgr, oracle, depth_gt, pidnet_flat, pidnet_depth, valid):
    rgb = cv2.resize(decoded_bgr, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(rgb, (0, 0), (319, 28), (15, 15, 15), -1)
    cv2.putText(rgb, "deployed JPEG input", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    disagreement = np.zeros((*oracle.shape, 3), dtype=np.uint8)
    disagreement[valid & (oracle == pidnet_depth)] = (55, 120, 55)
    disagreement[valid & oracle & (~pidnet_depth)] = (0, 0, 255)
    disagreement[valid & (~oracle) & pidnet_depth] = (0, 190, 255)
    disagreement_panel = cv2.resize(disagreement, (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(disagreement_panel, (0, 0), (319, 28), (15, 15, 15), -1)
    cv2.putText(
        disagreement_panel,
        "D error: red=miss amber=false stop",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    panels = (
        rgb,
        occupancy_panel(oracle, valid, "A Oracle control"),
        occupancy_panel(depth_gt, valid, "B GT + depth-lift"),
        occupancy_panel(pidnet_flat, valid, "C PIDNet + flat IPM"),
        occupancy_panel(pidnet_depth, valid, "D PIDNet + depth-lift"),
        disagreement_panel,
    )
    canvas = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write evidence image: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")

    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    phase5a = json.loads(PHASE5A_STATUS.read_text(encoding="utf-8"))
    trajectory_summary_path, trajectory = load_trajectory(phase5a)
    if args.max_frames:
        trajectory = trajectory[: args.max_frames]
    output = args.output or ROOT / "artifacts/phase5b_shadow" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()

    oracle_manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    oracle_archive_path = oracle_manifest_path.parent / oracle_manifest["archive"]
    oracle_archive = np.load(oracle_archive_path)
    raw_occupied = oracle_archive["raw_occupied"]
    sensor = phase3["sensor_geometry"]
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    map_x, map_y = build_ipm_remap(np, cv2, sensor, bev)
    ipm_valid = map_x >= 0.0
    candidate = CandidatePidnet(PIDNET_MODEL)
    evidence_indices = set(np.linspace(0, len(trajectory) - 1, min(18, len(trajectory)), dtype=int))

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    rows = []
    evidence = []
    try:
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom

        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        camera = UsdGeom.Camera.Define(stage, "/Phase5BShadowCamera")
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

        csv_path = output / "frames.csv"
        with csv_path.open("w", newline="", encoding="ascii") as target:
            writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for frame_id, frame in enumerate(trajectory):
                x, y, yaw = frame["x_m"], frame["y_m"], frame["yaw_rad"]
                ext = sensor["body_extrinsics"]
                eye = (
                    x + math.cos(yaw) * ext["forward_m"] - math.sin(yaw) * ext["left_m"],
                    y + math.sin(yaw) * ext["forward_m"] + math.cos(yaw) * ext["left_m"],
                    ext["height_m"],
                )
                transform.Set(camera_matrix(eye, yaw + ext["yaw_rad"], ext["pitch_rad"], Gf))
                rep.orchestrator.step()
                rgb = np.asarray(annotators["rgb"].get_data())[:, :, :3].astype(np.uint8)
                semantic_ids, _ = semantic_id_image(annotators["semantic"].get_data(), np)
                depth = np.asarray(annotators["depth"].get_data(), dtype=np.float32)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                candidate_classes, candidate_latency = candidate.infer(decoded_bgr)

                depth_started = time.perf_counter_ns()
                depth_gt_bev, depth_gt_valid = depth_lift_semantic_bev(
                    semantic_ids, depth, sensor, bev, np
                )
                candidate_ids = np.where(candidate_classes == 0, 0, 13).astype(np.uint8)
                candidate_depth_bev, candidate_depth_valid = depth_lift_semantic_bev(
                    candidate_ids, depth, sensor, bev, np
                )
                depth_latency = (time.perf_counter_ns() - depth_started) / 1e6
                oracle, oracle_valid = oracle_bev(
                    raw_occupied, oracle_manifest, (x, y, yaw), bev
                )
                candidate_flat_ids = cv2.remap(
                    candidate_classes,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                )
                candidates = {
                    "depth_gt": occupancy_from_semantic(depth_gt_bev),
                    "pidnet_flat": candidate_flat_ids != 0,
                    "pidnet_depth": occupancy_from_semantic(candidate_depth_bev),
                }
                common_valid = (
                    fixed_roi
                    & oracle_valid
                    & ipm_valid
                    & depth_gt_valid
                    & candidate_depth_valid
                )
                nearest = {
                    "oracle": nearest_center_obstacle(oracle, common_valid, bev),
                    **{
                        method: nearest_center_obstacle(value, common_valid, bev)
                        for method, value in candidates.items()
                    },
                }
                metrics = {
                    method: occupancy_metrics(oracle, value, common_valid)
                    for method, value in candidates.items()
                }
                stop = {
                    name: distance is not None and distance <= 0.65
                    for name, distance in nearest.items()
                }
                row = {
                    "source_frame_id": frame_id,
                    "scenario": frame["scenario"],
                    "scenario_step": frame["scenario_step"],
                    "simulation_time_s": frame["simulation_time_s"],
                    "x_m": x,
                    "y_m": y,
                    "yaw_rad": yaw,
                    "oracle_acceleration_mps2": frame["acceleration_mps2"],
                    "oracle_omega_radps": frame["omega_radps"],
                    "oracle_velocity_mps": frame["velocity_mps"],
                    "common_roi_coverage": float(common_valid.sum() / fixed_roi.sum()),
                    "depth_latency_ms": depth_latency,
                    "candidate_latency_ms": candidate_latency,
                    "candidate_valid": 1,
                    "nearest_oracle_m": nearest["oracle"],
                    "nearest_depth_gt_m": nearest["depth_gt"],
                    "nearest_pidnet_flat_m": nearest["pidnet_flat"],
                    "nearest_pidnet_depth_m": nearest["pidnet_depth"],
                    "oracle_stop": int(stop["oracle"]),
                    "depth_gt_stop": int(stop["depth_gt"]),
                    "pidnet_flat_stop": int(stop["pidnet_flat"]),
                    "pidnet_depth_stop": int(stop["pidnet_depth"]),
                }
                for method in METHODS:
                    row.update(
                        {f"{method}_{name}": value for name, value in metrics[method].items()}
                    )
                writer.writerow(row)
                rows.append(row)
                if frame_id in evidence_indices:
                    evidence_path = evidence_dir / f"frame_{frame_id:06d}.png"
                    write_evidence(
                        evidence_path,
                        decoded_bgr,
                        oracle,
                        candidates["depth_gt"],
                        candidates["pidnet_flat"],
                        candidates["pidnet_depth"],
                        common_valid,
                    )
                    evidence.append(
                        {
                            "source_frame_id": frame_id,
                            "path": str(evidence_path.relative_to(output)),
                            "sha256": sha256(evidence_path),
                        }
                    )
                if (frame_id + 1) % 100 == 0 or frame_id + 1 == len(trajectory):
                    print(
                        f"[Phase 5-B] frames={frame_id + 1}/{len(trajectory)} "
                        f"candidate={candidate_latency:.2f}ms coverage={row['common_roi_coverage']:.3f}"
                    )
    except BaseException:
        app.close()
        raise

    thresholds = phase3["phase4_perception_gate"]
    summary = {
        "schema_version": "phase5b-shadow-v1",
        "status": "shadow_evidence_complete",
        "control_authority": {
            "owner": "Phase 5-A frozen Oracle NMPC telemetry",
            "shadow_outputs_can_control": False,
            "control_output_declared_by_this_program": False,
        },
        "frame_count": len(rows),
        "minimum_gate_frames": thresholds["minimum_frames"],
        "synchronization": {
            "exact_frame_ratio": 1.0,
            "common_roi_coverage": aggregate(rows, "common_roi_coverage"),
        },
        "latency_ms": {
            "depth_lift": aggregate(rows, "depth_latency_ms"),
            "pidnet_total": aggregate(rows, "candidate_latency_ms"),
        },
        "candidate": {
            "id": "pidnet_s_cityscapes19_negative_control",
            "model_path": str(PIDNET_MODEL.relative_to(ROOT)),
            "model_sha256": sha256(PIDNET_MODEL),
            "providers": candidate.providers,
            "semantic_adaptation": "none",
            "valid_ratio": float(np.mean([row["candidate_valid"] for row in rows])),
            "eligible_to_control": False,
        },
        "gate_thresholds": {
            "false_free_rate_mean_max": thresholds["bc_false_free_rate_mean_max"],
            "false_occupied_rate_mean_max": thresholds["bc_false_occupied_rate_mean_max"],
            "free_iou_mean_min": thresholds["bc_free_iou_mean_min"],
            "occupied_iou_mean_min": thresholds["bc_occupied_iou_mean_min"],
            "latency_p95_ms_max": thresholds["latency_p95_ms_max"],
        },
        "methods": {
            method: method_summary(rows, method, {
                "false_free_rate_mean_max": thresholds["bc_false_free_rate_mean_max"],
                "false_occupied_rate_mean_max": thresholds["bc_false_occupied_rate_mean_max"],
                "free_iou_mean_min": thresholds["bc_free_iou_mean_min"],
                "occupied_iou_mean_min": thresholds["bc_occupied_iou_mean_min"],
            })
            for method in METHODS
        },
        "sources": {
            "phase5a_status": str(PHASE5A_STATUS.relative_to(ROOT)),
            "phase5a_status_sha256": sha256(PHASE5A_STATUS),
            "trajectory_summary": str(trajectory_summary_path.relative_to(ROOT)),
            "trajectory_summary_sha256": sha256(trajectory_summary_path),
            "oracle_manifest": str(oracle_manifest_path.relative_to(ROOT)),
            "oracle_manifest_sha256": sha256(oracle_manifest_path),
            "semantic_overlay": str(PHASE4_OVERLAY.relative_to(ROOT)),
            "semantic_overlay_sha256": sha256(PHASE4_OVERLAY),
        },
        "telemetry": "frames.csv",
        "telemetry_sha256": sha256(output / "frames.csv"),
        "evidence": evidence,
        "limitations": [
            "static warehouse scene only; no dynamic obstacle recall claim",
            "Oracle trajectory has no emergency-stop-positive frame; dangerous false-go counts are not recall evidence",
            "PIDNet is an unadapted Cityscapes negative control, not a warehouse candidate",
            "real camera distortion remains unfrozen",
        ],
    }
    candidate_gate = summary["methods"]["pidnet_flat"]["gate_passed"]
    depth_gate = summary["methods"]["depth_gt"]["gate_passed"]
    enough_frames = len(rows) >= thresholds["minimum_frames"]
    latency_ok = summary["latency_ms"]["pidnet_total"]["p95"] <= thresholds["latency_p95_ms_max"]
    oracle_stop_frames = sum(int(row["oracle_stop"]) for row in rows)
    summary["risk_decision_coverage"] = {
        "oracle_stop_frames": oracle_stop_frames,
        "oracle_go_frames": len(rows) - oracle_stop_frames,
        "positive_stop_frames_present": oracle_stop_frames > 0,
        "dangerous_false_go_claim_allowed": oracle_stop_frames > 0,
    }
    summary["verdict"] = {
        "evidence_volume_passed": enough_frames,
        "latency_passed": latency_ok,
        "depth_lift_gate_passed": depth_gate,
        "candidate_metric_gate_passed": candidate_gate,
        "candidate_gate_passed": bool(enough_frames and latency_ok and candidate_gate),
        "candidate_control_gate": "closed",
    }
    if args.max_frames is not None:
        summary["status"] = "smoke_only"
        summary["verdict"]["evidence_volume_passed"] = False
    elif not enough_frames:
        summary["status"] = "insufficient_evidence"
    elif not candidate_gate:
        summary["status"] = "shadow_evidence_complete_candidate_rejected"
    else:
        summary["status"] = "shadow_evidence_complete_candidate_passed_not_promoted"
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary["verdict"], indent=2))
    print(f"Phase 5-B artifacts: {output}")
    app.close()


if __name__ == "__main__":
    main()
