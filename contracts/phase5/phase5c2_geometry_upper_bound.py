#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from depth_bev_geometry import depth_lift_semantic_bev
from phase5b_shadow_replay import (
    aggregate,
    control_roi,
    load_trajectory,
    nearest_center_obstacle,
    occupancy_from_semantic,
    oracle_bev,
)
from phase5c_dynamic_upper_bound import (
    CSV_FIELDS,
    DYNAMIC_HALF_EXTENT_M,
    FRAME_COUNT,
    STOP_DISTANCE_M,
    dynamic_case,
    sha256,
    stop_metrics,
    world_from_ego,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
PHASE5A_STATUS = ROOT / "contracts/phase5/phase5a_status.json"
PHASE5C_STATUS = ROOT / "contracts/phase5/phase5c_status.json"


def dynamic_footprint_mask(bev, forward_m, left_m):
    rows, cols = bev["shape"][0], bev["shape"][1]
    row, col = np.indices((rows, cols), dtype=np.float32)
    meters = bev["meters_per_cell"]
    forward = (bev["ego_origin_cell"][0] - row) * meters
    left = (bev["ego_origin_cell"][1] - col) * meters
    half_cell = meters / 2.0
    return (
        (np.abs(forward - forward_m) <= DYNAMIC_HALF_EXTENT_M + half_cell)
        & (np.abs(left - left_m) <= DYNAMIC_HALF_EXTENT_M + half_cell)
    )


def occupancy_metrics_contract(oracle, candidate, valid):
    oracle = np.asarray(oracle, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    oracle_occupied = oracle & valid
    candidate_occupied = candidate & valid
    oracle_free = (~oracle) & valid
    candidate_free = (~candidate) & valid
    true_positive = int(np.count_nonzero(oracle_occupied & candidate_occupied))
    true_negative = int(np.count_nonzero(oracle_free & candidate_free))
    false_positive = int(np.count_nonzero(oracle_free & candidate_occupied))
    false_negative = int(np.count_nonzero(oracle_occupied & candidate_free))
    valid_count = true_positive + true_negative + false_positive + false_negative
    occupied_union = true_positive + false_positive + false_negative
    free_union = true_negative + false_positive + false_negative
    return {
        "occupied_iou": true_positive / occupied_union if occupied_union else 1.0,
        "free_iou": true_negative / free_union if free_union else 1.0,
        "false_free_rate": false_negative / valid_count if valid_count else 0.0,
        "false_occupied_rate": false_positive / valid_count if valid_count else 0.0,
        "agreement": (true_positive + true_negative) / valid_count if valid_count else 0.0,
        "true_positive_count": true_positive,
        "true_negative_count": true_negative,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
    }


def method_summary_contract(rows, thresholds):
    metrics = {
        name: aggregate(rows, f"depth_gt_{name}")
        for name in (
            "occupied_iou",
            "free_iou",
            "false_free_rate",
            "false_occupied_rate",
            "agreement",
        )
    }
    counts = {
        name: sum(int(row[f"depth_gt_{name}_count"]) for row in rows)
        for name in ("true_positive", "true_negative", "false_positive", "false_negative")
    }
    tp = counts["true_positive"]
    tn = counts["true_negative"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    valid_count = tp + tn + fp + fn
    metrics["micro_confusion"] = {
        **counts,
        "occupied_iou": tp / max(tp + fp + fn, 1),
        "free_iou": tn / max(tn + fp + fn, 1),
        "false_free_rate": fn / max(valid_count, 1),
        "false_occupied_rate": fp / max(valid_count, 1),
        "conditional_obstacle_miss_rate": fn / max(tp + fn, 1),
        "conditional_free_false_alarm_rate": fp / max(tn + fp, 1),
    }
    metrics["gate_passed"] = bool(
        metrics["false_free_rate"]["mean"] <= thresholds["false_free_rate_mean_max"]
        and metrics["false_occupied_rate"]["mean"] <= thresholds["false_occupied_rate_mean_max"]
        and metrics["free_iou"]["mean"] >= thresholds["free_iou_mean_min"]
        and metrics["occupied_iou"]["mean"] >= thresholds["occupied_iou_mean_min"]
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame_count = args.max_frames or FRAME_COUNT
    if not 1 <= frame_count <= FRAME_COUNT:
        raise SystemExit(f"--max-frames must be in [1, {FRAME_COUNT}]")
    oracle_manifest_path = args.oracle_manifest.resolve()
    if not oracle_manifest_path.is_file():
        raise SystemExit(f"perception Oracle manifest is missing: {oracle_manifest_path}")

    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    phase5a = json.loads(PHASE5A_STATUS.read_text(encoding="utf-8"))
    trajectory_summary_path, trajectory = load_trajectory(phase5a)
    trajectory = trajectory[:frame_count]
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    if oracle_manifest["status"] != "perception_scoring_only":
        raise SystemExit("Phase 5-C2 requires a perception-only Oracle")
    oracle_archive_path = oracle_manifest_path.parent / oracle_manifest["archive"]
    oracle_archive = np.load(oracle_archive_path)
    perception_occupied = oracle_archive["perception_occupied"]

    output = args.output or ROOT / "artifacts/phase5c2_geometry" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    sensor = phase3["sensor_geometry"]
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    evidence_indices = set(
        np.linspace(0, frame_count - 1, min(20, frame_count), dtype=int)
    )

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    rows = []
    evidence = []
    try:
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom

        from capture_smoke_dataset import camera_matrix, semantic_id_image

        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        obstacle = UsdGeom.Cube.Define(stage, "/Phase5C2DynamicObstacle")
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        obstacle_translate = UsdGeom.Xformable(obstacle).AddTranslateOp()
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")

        camera = UsdGeom.Camera.Define(stage, "/Phase5C2DynamicCamera")
        width, height = sensor["image_size"]
        horizontal_aperture = 20.955
        vertical_aperture = sensor["intrinsics"]["fx"] * horizontal_aperture * height / (
            sensor["intrinsics"]["fy"] * width
        )
        focal_length = sensor["intrinsics"]["fx"] * horizontal_aperture / width
        camera.CreateHorizontalApertureAttr(horizontal_aperture)
        camera.CreateVerticalApertureAttr(vertical_aperture)
        camera.CreateFocalLengthAttr(focal_length)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera_transform = UsdGeom.Xformable(camera).AddTransformOp()
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
                mode, dynamic_forward, dynamic_left = dynamic_case(frame_id)
                obstacle_x, obstacle_y = world_from_ego(
                    x, y, yaw, dynamic_forward, dynamic_left
                )
                obstacle_translate.Set(
                    Gf.Vec3d(obstacle_x, obstacle_y, DYNAMIC_HALF_EXTENT_M)
                )
                ext = sensor["body_extrinsics"]
                eye = (
                    x + math.cos(yaw) * ext["forward_m"] - math.sin(yaw) * ext["left_m"],
                    y + math.sin(yaw) * ext["forward_m"] + math.cos(yaw) * ext["left_m"],
                    ext["height_m"],
                )
                camera_transform.Set(
                    camera_matrix(eye, yaw + ext["yaw_rad"], ext["pitch_rad"], Gf)
                )
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

                started = time.perf_counter_ns()
                depth_bev, depth_valid = depth_lift_semantic_bev(
                    semantic_ids, depth, sensor, bev
                )
                depth_latency = (time.perf_counter_ns() - started) / 1e6
                depth_occupied = occupancy_from_semantic(depth_bev)
                oracle, oracle_valid = oracle_bev(
                    perception_occupied, oracle_manifest, (x, y, yaw), bev
                )
                if dynamic_forward > 0.0:
                    oracle |= dynamic_footprint_mask(
                        bev, dynamic_forward, dynamic_left
                    )
                common_valid = fixed_roi & oracle_valid & depth_valid
                nearest_oracle = nearest_center_obstacle(
                    oracle, fixed_roi & oracle_valid, bev
                )
                nearest_depth = nearest_center_obstacle(
                    depth_occupied, fixed_roi & depth_valid, bev
                )
                oracle_stop = nearest_oracle is not None and nearest_oracle <= STOP_DISTANCE_M
                depth_stop = nearest_depth is not None and nearest_depth <= STOP_DISTANCE_M
                metrics = occupancy_metrics_contract(oracle, depth_occupied, common_valid)
                row = {
                    "source_frame_id": frame_id,
                    "scenario": frame["scenario"],
                    "scenario_step": frame["scenario_step"],
                    "x_m": x,
                    "y_m": y,
                    "yaw_rad": yaw,
                    "dynamic_mode": mode,
                    "dynamic_forward_m": dynamic_forward,
                    "dynamic_left_m": dynamic_left,
                    "common_roi_coverage": float(common_valid.sum() / fixed_roi.sum()),
                    "depth_latency_ms": depth_latency,
                    "nearest_oracle_m": nearest_oracle,
                    "nearest_depth_gt_m": nearest_depth,
                    "oracle_stop": int(oracle_stop),
                    "depth_gt_stop": int(depth_stop),
                    **{f"depth_gt_{name}": value for name, value in metrics.items()},
                }
                writer.writerow(row)
                rows.append(row)
                if frame_id in evidence_indices:
                    evidence_path = evidence_dir / f"frame_{frame_id:06d}_{mode}.png"
                    write_evidence(
                        evidence_path,
                        decoded_bgr,
                        oracle,
                        depth_occupied,
                        common_valid,
                        mode,
                    )
                    evidence.append(
                        {
                            "source_frame_id": frame_id,
                            "dynamic_mode": mode,
                            "path": str(evidence_path.relative_to(output)),
                            "sha256": sha256(evidence_path),
                        }
                    )
                if (frame_id + 1) % 100 == 0 or frame_id + 1 == frame_count:
                    print(
                        f"[Phase 5-C2 geometry] frames={frame_id + 1}/{frame_count} "
                        f"mode={mode} oracle_stop={int(oracle_stop)} depth_stop={int(depth_stop)}"
                    )
    except BaseException:
        app.close()
        raise

    thresholds = phase3["phase4_perception_gate"]
    method = method_summary_contract(
        rows,
        {
            "false_free_rate_mean_max": thresholds["bc_false_free_rate_mean_max"],
            "false_occupied_rate_mean_max": thresholds["bc_false_occupied_rate_mean_max"],
            "free_iou_mean_min": thresholds["bc_free_iou_mean_min"],
            "occupied_iou_mean_min": thresholds["bc_occupied_iou_mean_min"],
        },
    )
    stopping = stop_metrics(rows)
    stop_gate_passed = stopping["stop_recall"] >= 0.95 and stopping["go_specificity"] >= 0.95
    upper_bound_passed = bool(
        frame_count >= FRAME_COUNT and method["gate_passed"] and stop_gate_passed
    )
    summary = {
        "schema_version": "phase5c2-geometry-upper-bound-v1",
        "status": (
            "geometry_upper_bound_passed_model_training_allowed"
            if upper_bound_passed
            else "geometry_upper_bound_failed_model_training_blocked"
        ),
        "frame_count": frame_count,
        "control_authority": {
            "owner": "Phase 5-A USD Oracle NMPC",
            "control_output_declared": False,
            "dynamic_obstacle_affects_control": False,
            "perception_oracle_controls_vehicle": False,
        },
        "geometry_corrections": {
            "depth_height_filter_m": [0.02, 0.35],
            "free_floor_height_filter_m": [-0.05, 0.08],
            "oracle_geometry": "exact USD mesh face slice; Phase 5-A control AABB unchanged",
            "error_rate_denominator": "all valid ROI cells, as frozen by Phase 2/4",
        },
        "dynamic_schedule": {
            "center_stop_frames_expected": frame_count * 4 // 10,
            "go_frames_expected": frame_count * 6 // 10,
            "stop_distance_m": STOP_DISTANCE_M,
            "obstacle_size_m": DYNAMIC_HALF_EXTENT_M * 2.0,
        },
        "synchronization": {
            "exact_frame_ratio": 1.0,
            "common_roi_coverage": aggregate(rows, "common_roi_coverage"),
        },
        "depth_latency_ms": aggregate(rows, "depth_latency_ms"),
        "gt_depth_lift": method,
        "stop_decision": stopping,
        "gate": {
            "minimum_frames": FRAME_COUNT,
            "stop_recall_min": 0.95,
            "go_specificity_min": 0.95,
            "perception_metrics_passed": method["gate_passed"],
            "stop_decision_passed": stop_gate_passed,
            "upper_bound_passed": upper_bound_passed,
            "model_training_allowed": upper_bound_passed,
            "control_promotion_allowed": False,
        },
        "sources": {
            "phase5c_status": str(PHASE5C_STATUS.relative_to(ROOT)),
            "phase5c_status_sha256": sha256(PHASE5C_STATUS),
            "trajectory_summary": str(trajectory_summary_path.relative_to(ROOT)),
            "trajectory_summary_sha256": sha256(trajectory_summary_path),
            "semantic_overlay": str(PHASE4_OVERLAY.relative_to(ROOT)),
            "semantic_overlay_sha256": sha256(PHASE4_OVERLAY),
            "perception_oracle_manifest": str(oracle_manifest_path.relative_to(ROOT)),
            "perception_oracle_manifest_sha256": sha256(oracle_manifest_path),
            "perception_oracle_archive_sha256": sha256(oracle_archive_path),
        },
        "telemetry": "frames.csv",
        "telemetry_sha256": sha256(output / "frames.csv"),
        "evidence": evidence,
        "next_action": (
            "capture disjoint warehouse train/validation data and adapt the semantic model"
            if upper_bound_passed
            else "continue white-box geometry repair; model training remains blocked"
        ),
    }
    if args.max_frames is not None:
        summary["status"] = "smoke_only"
        summary["gate"]["upper_bound_passed"] = False
        summary["gate"]["model_training_allowed"] = False
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary["gate"], indent=2))
    print(f"Phase 5-C2 geometry artifacts: {output}")
    app.close()


if __name__ == "__main__":
    main()
