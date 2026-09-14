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

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
PHASE5A_STATUS = ROOT / "contracts/phase5/phase5a_status.json"
PHASE4_DIR = ROOT / "contracts/phase4"
PHASE5_DIR = ROOT / "contracts/phase5"
sys.path.insert(0, str(PHASE4_DIR))
sys.path.insert(0, str(PHASE5_DIR))
from capture_smoke_dataset import camera_matrix, depth_lift_semantic_bev, semantic_id_image  # noqa: E402
from phase5b_shadow_replay import (  # noqa: E402
    aggregate,
    control_roi,
    load_trajectory,
    method_summary,
    nearest_center_obstacle,
    occupancy_from_semantic,
    occupancy_metrics,
    occupancy_panel,
    oracle_bev,
)

FRAME_COUNT = 1000
STOP_DISTANCE_M = 0.65
DYNAMIC_HALF_EXTENT_M = 0.225
CSV_FIELDS = [
    "source_frame_id",
    "scenario",
    "scenario_step",
    "x_m",
    "y_m",
    "yaw_rad",
    "dynamic_mode",
    "dynamic_forward_m",
    "dynamic_left_m",
    "common_roi_coverage",
    "depth_latency_ms",
    "nearest_oracle_m",
    "nearest_depth_gt_m",
    "oracle_stop",
    "depth_gt_stop",
    "depth_gt_occupied_iou",
    "depth_gt_free_iou",
    "depth_gt_false_free_rate",
    "depth_gt_false_occupied_rate",
    "depth_gt_agreement",
    "depth_gt_true_positive_count",
    "depth_gt_true_negative_count",
    "depth_gt_false_positive_count",
    "depth_gt_false_negative_count",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dynamic_case(frame_id):
    phase = frame_id % 10
    if phase < 4:
        return "center_stop", 0.50 + phase * 0.04, 0.0
    if phase == 4:
        return "side_go_left", 0.55, 1.05
    if phase == 5:
        return "side_go_right", 0.55, -1.05
    if phase < 8:
        return "far_go", 1.20 + (phase - 6) * 0.30, 0.0
    return "absent_go", -5.0, 0.0


def dynamic_mask(bev, forward_m, left_m):
    rows, cols = bev["shape"][0], bev["shape"][1]
    row, col = np.indices((rows, cols), dtype=np.float32)
    forward = (bev["ego_origin_cell"][0] - row) * bev["meters_per_cell"]
    left = (bev["ego_origin_cell"][1] - col) * bev["meters_per_cell"]
    return (
        (np.abs(forward - forward_m) <= DYNAMIC_HALF_EXTENT_M)
        & (np.abs(left - left_m) <= DYNAMIC_HALF_EXTENT_M)
    )


def world_from_ego(x, y, yaw, forward, left):
    return (
        x + forward * math.cos(yaw) - left * math.sin(yaw),
        y + forward * math.sin(yaw) + left * math.cos(yaw),
    )


def write_evidence(path, decoded_bgr, oracle, depth_gt, valid, mode):
    rgb = cv2.resize(decoded_bgr, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(rgb, (0, 0), (319, 28), (15, 15, 15), -1)
    cv2.putText(rgb, mode, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    error = np.zeros((*oracle.shape, 3), dtype=np.uint8)
    error[valid & (oracle == depth_gt)] = (55, 120, 55)
    error[valid & oracle & (~depth_gt)] = (0, 0, 255)
    error[valid & (~oracle) & depth_gt] = (0, 190, 255)
    error = cv2.resize(error, (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(error, (0, 0), (319, 28), (15, 15, 15), -1)
    cv2.putText(error, "red=miss amber=false stop", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.hstack(
        (
            rgb,
            occupancy_panel(oracle, valid, "Oracle + dynamic"),
            occupancy_panel(depth_gt, valid, "GT semantic + depth"),
            error,
        )
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write evidence: {path}")


def stop_metrics(rows):
    true_stop = sum(int(row["oracle_stop"] and row["depth_gt_stop"]) for row in rows)
    missed_stop = sum(int(row["oracle_stop"] and not row["depth_gt_stop"]) for row in rows)
    true_go = sum(int(not row["oracle_stop"] and not row["depth_gt_stop"]) for row in rows)
    false_stop = sum(int(not row["oracle_stop"] and row["depth_gt_stop"]) for row in rows)
    return {
        "true_stop": true_stop,
        "missed_stop": missed_stop,
        "true_go": true_go,
        "false_stop": false_stop,
        "stop_recall": true_stop / max(true_stop + missed_stop, 1),
        "go_specificity": true_go / max(true_go + false_stop, 1),
        "stop_precision": true_stop / max(true_stop + false_stop, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frame_count = args.max_frames or FRAME_COUNT
    if not 1 <= frame_count <= FRAME_COUNT:
        raise SystemExit(f"--max-frames must be in [1, {FRAME_COUNT}]")

    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    phase5a = json.loads(PHASE5A_STATUS.read_text(encoding="utf-8"))
    trajectory_summary_path, trajectory = load_trajectory(phase5a)
    trajectory = trajectory[:frame_count]
    output = args.output or ROOT / "artifacts/phase5c_upper_bound" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()

    oracle_manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    oracle_archive = np.load(oracle_manifest_path.parent / oracle_manifest["archive"])
    raw_occupied = oracle_archive["raw_occupied"]
    sensor = phase3["sensor_geometry"]
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    evidence_indices = set(np.linspace(0, frame_count - 1, min(20, frame_count), dtype=int))

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

        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        obstacle = UsdGeom.Cube.Define(stage, "/Phase5CDynamicObstacle")
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        obstacle_translate = UsdGeom.Xformable(obstacle).AddTranslateOp()
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")

        camera = UsdGeom.Camera.Define(stage, "/Phase5CDynamicCamera")
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
                obstacle_translate.Set(Gf.Vec3d(obstacle_x, obstacle_y, DYNAMIC_HALF_EXTENT_M))
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
                    semantic_ids, depth, sensor, bev, np
                )
                depth_latency = (time.perf_counter_ns() - started) / 1e6
                depth_occupied = occupancy_from_semantic(depth_bev)
                oracle, oracle_valid = oracle_bev(
                    raw_occupied, oracle_manifest, (x, y, yaw), bev
                )
                if dynamic_forward > 0.0:
                    oracle |= dynamic_mask(bev, dynamic_forward, dynamic_left)
                common_valid = fixed_roi & oracle_valid & depth_valid
                nearest_oracle = nearest_center_obstacle(
                    oracle, fixed_roi & oracle_valid, bev
                )
                nearest_depth = nearest_center_obstacle(
                    depth_occupied, fixed_roi & depth_valid, bev
                )
                oracle_stop = nearest_oracle is not None and nearest_oracle <= STOP_DISTANCE_M
                depth_stop = nearest_depth is not None and nearest_depth <= STOP_DISTANCE_M
                metrics = occupancy_metrics(oracle, depth_occupied, common_valid)
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
                        f"[Phase 5-C upper bound] frames={frame_id + 1}/{frame_count} "
                        f"mode={mode} oracle_stop={int(oracle_stop)} depth_stop={int(depth_stop)}"
                    )
    except BaseException:
        app.close()
        raise

    thresholds = phase3["phase4_perception_gate"]
    method = method_summary(
        rows,
        "depth_gt",
        {
            "false_free_rate_mean_max": thresholds["bc_false_free_rate_mean_max"],
            "false_occupied_rate_mean_max": thresholds["bc_false_occupied_rate_mean_max"],
            "free_iou_mean_min": thresholds["bc_free_iou_mean_min"],
            "occupied_iou_mean_min": thresholds["bc_occupied_iou_mean_min"],
        },
    )
    stopping = stop_metrics(rows)
    volume_passed = frame_count >= FRAME_COUNT
    stop_gate_passed = stopping["stop_recall"] >= 0.95 and stopping["go_specificity"] >= 0.95
    upper_bound_passed = bool(volume_passed and method["gate_passed"] and stop_gate_passed)
    summary = {
        "schema_version": "phase5c-dynamic-upper-bound-v1",
        "status": (
            "gt_upper_bound_passed_model_training_allowed"
            if upper_bound_passed
            else "gt_upper_bound_failed_model_training_blocked"
        ),
        "frame_count": frame_count,
        "control_authority": {
            "mode": "counterfactual shadow replay",
            "control_output_declared": False,
            "dynamic_obstacle_affects_control": False,
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
        },
        "sources": {
            "phase5a_status": str(PHASE5A_STATUS.relative_to(ROOT)),
            "phase5a_status_sha256": sha256(PHASE5A_STATUS),
            "trajectory_summary": str(trajectory_summary_path.relative_to(ROOT)),
            "trajectory_summary_sha256": sha256(trajectory_summary_path),
            "semantic_overlay": str(PHASE4_OVERLAY.relative_to(ROOT)),
            "semantic_overlay_sha256": sha256(PHASE4_OVERLAY),
            "oracle_manifest": str(oracle_manifest_path.relative_to(ROOT)),
            "oracle_manifest_sha256": sha256(oracle_manifest_path),
        },
        "telemetry": "frames.csv",
        "telemetry_sha256": sha256(output / "frames.csv"),
        "evidence": evidence,
        "next_action": (
            "capture disjoint train set and fine-tune warehouse_nav14 model"
            if upper_bound_passed
            else "repair camera/depth-to-BEV geometry before training a warehouse model"
        ),
    }
    if args.max_frames is not None:
        summary["status"] = "smoke_only"
        summary["gate"]["upper_bound_passed"] = False
        summary["gate"]["model_training_allowed"] = False
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary["gate"], indent=2))
    print(f"Phase 5-C upper-bound artifacts: {output}")
    app.close()


if __name__ == "__main__":
    main()
