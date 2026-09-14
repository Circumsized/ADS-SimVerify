#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from depth_bev_geometry import (
    FLOOR_MIN_HEIGHT_M,
    OBSTACLE_MAX_HEIGHT_M,
    camera_depth_to_body,
    depth_lift_semantic_bev,
    one_hot,
)
from phase5b_shadow_replay import (
    aggregate,
    control_roi,
    load_trajectory,
    nearest_center_obstacle,
    occupancy_from_semantic,
    occupancy_panel,
    oracle_bev,
)
from phase5c2_geometry_upper_bound import (
    dynamic_footprint_mask,
    occupancy_metrics_contract,
)
from phase5c_dynamic_upper_bound import (
    DYNAMIC_HALF_EXTENT_M,
    FRAME_COUNT,
    STOP_DISTANCE_M,
    dynamic_case,
    sha256,
    stop_metrics,
    world_from_ego,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
PHASE5A_STATUS = ROOT / "contracts/phase5/phase5a_status.json"
PHASE5C2_STATUS = ROOT / "contracts/phase5/phase5c2_status.json"
MODEL_MANIFEST = ROOT / "model/warehouse_nav14_candidate.json"
METHODS = ("depth_gt", "candidate")
BASE_FIELDS = [
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
    "candidate_valid_ratio",
    "teacher_depth_latency_ms",
    "candidate_inference_latency_ms",
    "candidate_total_latency_ms",
    "nearest_oracle_m",
    "nearest_depth_gt_m",
    "nearest_candidate_m",
    "oracle_stop",
    "depth_gt_stop",
    "candidate_stop",
]
METRIC_NAMES = (
    "occupied_iou",
    "free_iou",
    "false_free_rate",
    "false_occupied_rate",
    "agreement",
    "true_positive_count",
    "true_negative_count",
    "false_positive_count",
    "false_negative_count",
)
CSV_FIELDS = BASE_FIELDS + [f"{method}_{name}" for method in METHODS for name in METRIC_NAMES]


class WarehouseCandidate:
    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def infer(self, decoded_bgr):
        started = time.perf_counter_ns()
        image = cv2.resize(decoded_bgr, (320, 240), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        tensor = np.transpose(image, (2, 0, 1))[None]
        logits = self.session.run(["output"], {"input": tensor})[0][0]
        best = np.argmax(logits, axis=0).astype(np.uint8)
        maximum = np.max(logits, axis=0, keepdims=True)
        confidence = 1.0 / np.exp(logits - maximum).sum(axis=0)
        best[confidence < 0.50] = 13
        class_map = cv2.resize(
            best,
            (decoded_bgr.shape[1], decoded_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return class_map, (time.perf_counter_ns() - started) / 1e6


def candidate_depth_lift(class_ids, depth, sensor, bev):
    forward, left, point_height, valid = camera_depth_to_body(class_ids, depth, sensor)
    geometry_supported = (
        (point_height >= FLOOR_MIN_HEIGHT_M)
        & (point_height <= OBSTACLE_MAX_HEIGHT_M)
    )
    valid &= geometry_supported
    rows, cols = bev["shape"][0], bev["shape"][1]
    meters = bev["meters_per_cell"]
    cell_row = np.rint(bev["ego_origin_cell"][0] - forward / meters).astype(np.int32)
    cell_col = np.rint(bev["ego_origin_cell"][1] - left / meters).astype(np.int32)
    valid &= (
        (cell_row >= 0)
        & (cell_row < rows)
        & (cell_col >= 0)
        & (cell_col < cols)
    )
    result = np.full((rows, cols), 13, dtype=np.uint8)
    observed = np.zeros((rows, cols), dtype=bool)
    for class_index in [13, 0, 1, *range(2, 13)]:
        selected = valid & (class_ids == class_index)
        result[cell_row[selected], cell_col[selected]] = class_index
        observed[cell_row[selected], cell_col[selected]] = True
    return one_hot(result), observed


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
    metrics["gate_passed"] = bool(
        metrics["false_free_rate"]["mean"] <= thresholds["false_free_rate_mean_max"]
        and metrics["false_occupied_rate"]["mean"] <= thresholds["false_occupied_rate_mean_max"]
        and metrics["free_iou"]["mean"] >= thresholds["free_iou_mean_min"]
        and metrics["occupied_iou"]["mean"] >= thresholds["occupied_iou_mean_min"]
    )
    return metrics


def stop_summary(rows, method):
    renamed = [
        {
            "oracle_stop": bool(row["oracle_stop"]),
            "depth_gt_stop": bool(row[f"{method}_stop"]),
        }
        for row in rows
    ]
    return stop_metrics(renamed)


def write_evidence(path, decoded_bgr, oracle, teacher, candidate, valid, mode):
    rgb = cv2.resize(decoded_bgr, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(rgb, (0, 0), (319, 28), (15, 15, 15), -1)
    cv2.putText(rgb, mode, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    error = np.zeros((*oracle.shape, 3), dtype=np.uint8)
    error[valid & (oracle == candidate)] = (55, 120, 55)
    error[valid & oracle & (~candidate)] = (0, 0, 255)
    error[valid & (~oracle) & candidate] = (0, 190, 255)
    error = cv2.resize(error, (320, 240), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack(
        (
            rgb,
            occupancy_panel(oracle, valid, "Oracle"),
            occupancy_panel(teacher, valid, "GT teacher"),
            occupancy_panel(candidate, valid, "warehouse candidate"),
            error,
        )
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write evidence: {path}")


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
    phase5c2 = json.loads(PHASE5C2_STATUS.read_text(encoding="utf-8"))
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    model_path = ROOT / model_manifest["model"]
    if sha256(model_path) != model_manifest["model_sha256"]:
        raise SystemExit("warehouse candidate hash differs from its manifest")
    _, trajectory = load_trajectory(phase5a)
    trajectory = trajectory[:frame_count]
    oracle_manifest_path = ROOT / phase5c2["perception_oracle"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    oracle_archive = np.load(ROOT / phase5c2["perception_oracle"]["archive"]["path"])
    perception_occupied = oracle_archive["perception_occupied"]
    output = args.output or ROOT / "artifacts/phase5c3_shadow" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir()
    sensor = phase3["sensor_geometry"]
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    candidate_model = WarehouseCandidate(model_path)
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
        from capture_smoke_dataset import camera_matrix, semantic_id_image

        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        obstacle = UsdGeom.Cube.Define(stage, "/Phase5C3GateDynamicObstacle")
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        obstacle_translate = UsdGeom.Xformable(obstacle).AddTranslateOp()
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
        camera = UsdGeom.Camera.Define(stage, "/Phase5C3GateCamera")
        width, height = sensor["image_size"]
        aperture = 20.955
        vertical_aperture = sensor["intrinsics"]["fx"] * aperture * height / (
            sensor["intrinsics"]["fy"] * width
        )
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(vertical_aperture)
        camera.CreateFocalLengthAttr(sensor["intrinsics"]["fx"] * aperture / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera_transform = UsdGeom.Xformable(camera).AddTransformOp()
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        annotators = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "semantic": rep.AnnotatorRegistry.get_annotator(
                "semantic_segmentation", init_params={"colorize": False}
            ),
            "depth": rep.AnnotatorRegistry.get_annotator("distance_to_image_plane"),
        }
        for annotator in annotators.values():
            annotator.attach([product])

        with (output / "frames.csv").open("w", newline="", encoding="ascii") as target:
            writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for frame_id, frame in enumerate(trajectory):
                x, y, yaw = frame["x_m"], frame["y_m"], frame["yaw_rad"]
                mode, dynamic_forward, dynamic_left = dynamic_case(frame_id)
                object_x, object_y = world_from_ego(x, y, yaw, dynamic_forward, dynamic_left)
                obstacle_translate.Set(Gf.Vec3d(object_x, object_y, DYNAMIC_HALF_EXTENT_M))
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
                    ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                candidate_ids, inference_latency = candidate_model.infer(decoded_bgr)
                started = time.perf_counter_ns()
                teacher_bev, teacher_valid = depth_lift_semantic_bev(
                    semantic_ids, depth, sensor, bev
                )
                teacher_latency = (time.perf_counter_ns() - started) / 1e6
                started = time.perf_counter_ns()
                candidate_bev, candidate_valid = candidate_depth_lift(
                    candidate_ids, depth, sensor, bev
                )
                candidate_total_latency = inference_latency + (
                    time.perf_counter_ns() - started
                ) / 1e6
                teacher_occupied = occupancy_from_semantic(teacher_bev)
                candidate_occupied = occupancy_from_semantic(candidate_bev)
                oracle, oracle_valid = oracle_bev(
                    perception_occupied, oracle_manifest, (x, y, yaw), bev
                )
                if dynamic_forward > 0.0:
                    oracle |= dynamic_footprint_mask(bev, dynamic_forward, dynamic_left)
                common_valid = fixed_roi & oracle_valid & teacher_valid & candidate_valid
                teacher_roi = fixed_roi & oracle_valid & teacher_valid
                candidate_valid_ratio = float(common_valid.sum() / max(teacher_roi.sum(), 1))
                occupied = {"depth_gt": teacher_occupied, "candidate": candidate_occupied}
                metrics = {
                    name: occupancy_metrics_contract(oracle, value, common_valid)
                    for name, value in occupied.items()
                }
                nearest = {
                    "oracle": nearest_center_obstacle(oracle, fixed_roi & oracle_valid, bev),
                    "depth_gt": nearest_center_obstacle(
                        teacher_occupied, fixed_roi & teacher_valid, bev
                    ),
                    "candidate": nearest_center_obstacle(
                        candidate_occupied, fixed_roi & candidate_valid, bev
                    ),
                }
                stops = {
                    name: distance is not None and distance <= STOP_DISTANCE_M
                    for name, distance in nearest.items()
                }
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
                    "candidate_valid_ratio": candidate_valid_ratio,
                    "teacher_depth_latency_ms": teacher_latency,
                    "candidate_inference_latency_ms": inference_latency,
                    "candidate_total_latency_ms": candidate_total_latency,
                    "nearest_oracle_m": nearest["oracle"],
                    "nearest_depth_gt_m": nearest["depth_gt"],
                    "nearest_candidate_m": nearest["candidate"],
                    "oracle_stop": int(stops["oracle"]),
                    "depth_gt_stop": int(stops["depth_gt"]),
                    "candidate_stop": int(stops["candidate"]),
                }
                for method in METHODS:
                    row.update({f"{method}_{name}": value for name, value in metrics[method].items()})
                writer.writerow(row)
                rows.append(row)
                if frame_id in evidence_indices:
                    evidence_path = evidence_dir / f"frame_{frame_id:06d}_{mode}.png"
                    write_evidence(
                        evidence_path,
                        decoded_bgr,
                        oracle,
                        teacher_occupied,
                        candidate_occupied,
                        common_valid,
                        mode,
                    )
                    evidence.append(
                        {
                            "source_frame_id": frame_id,
                            "path": str(evidence_path.relative_to(output)),
                            "sha256": sha256(evidence_path),
                        }
                    )
                if (frame_id + 1) % 100 == 0 or frame_id + 1 == frame_count:
                    print(
                        f"[Phase 5-C3 shadow] frames={frame_id + 1}/{frame_count} "
                        f"candidate={candidate_total_latency:.2f}ms stop={int(stops['candidate'])}"
                    )
    except BaseException:
        app.close()
        raise

    frozen = phase3["phase4_perception_gate"]
    thresholds = {
        "false_free_rate_mean_max": frozen["bc_false_free_rate_mean_max"],
        "false_occupied_rate_mean_max": frozen["bc_false_occupied_rate_mean_max"],
        "free_iou_mean_min": frozen["bc_free_iou_mean_min"],
        "occupied_iou_mean_min": frozen["bc_occupied_iou_mean_min"],
    }
    methods = {name: method_summary(rows, name, thresholds) for name in METHODS}
    stopping = {name: stop_summary(rows, name) for name in METHODS}
    valid_ratio = aggregate(rows, "candidate_valid_ratio")
    candidate_latency = aggregate(rows, "candidate_total_latency_ms")
    gate_passed = bool(
        frame_count >= FRAME_COUNT
        and methods["depth_gt"]["gate_passed"]
        and methods["candidate"]["gate_passed"]
        and stopping["candidate"]["stop_recall"] >= 0.95
        and stopping["candidate"]["go_specificity"] >= 0.95
        and valid_ratio["mean"] >= 0.99
        and candidate_latency["p95"] <= frozen["latency_p95_ms_max"]
    )
    summary = {
        "schema_version": "phase5c3-candidate-shadow-v1",
        "status": "candidate_shadow_passed" if gate_passed else "candidate_shadow_rejected",
        "frame_count": frame_count,
        "control_authority": {
            "owner": "Phase 5-A USD Oracle NMPC",
            "candidate_controls_vehicle": False,
            "control_output_declared": False,
        },
        "model": {
            "path": str(model_path.relative_to(ROOT)),
            "sha256": sha256(model_path),
            "manifest": str(MODEL_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha256(MODEL_MANIFEST),
        },
        "synchronization": {
            "exact_frame_ratio": 1.0,
            "common_roi_coverage": aggregate(rows, "common_roi_coverage"),
            "candidate_valid_ratio": valid_ratio,
        },
        "latency_ms": {
            "candidate_inference": aggregate(rows, "candidate_inference_latency_ms"),
            "candidate_total": candidate_latency,
        },
        "methods": methods,
        "stop_decision": stopping,
        "gate": {
            "candidate_metric_passed": methods["candidate"]["gate_passed"],
            "candidate_dynamic_passed": stopping["candidate"]["stop_recall"] >= 0.95
            and stopping["candidate"]["go_specificity"] >= 0.95,
            "candidate_validity_passed": valid_ratio["mean"] >= 0.99,
            "candidate_latency_passed": candidate_latency["p95"] <= frozen["latency_p95_ms_max"],
            "shadow_gate_passed": gate_passed,
            "control_promotion_allowed": False,
        },
        "telemetry": "frames.csv",
        "telemetry_sha256": sha256(output / "frames.csv"),
        "evidence": evidence,
    }
    if args.max_frames is not None:
        summary["status"] = "smoke_only"
        summary["gate"]["shadow_gate_passed"] = False
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary["gate"], indent=2))
    print(f"Phase 5-C3 shadow artifacts: {output}")
    app.close()


if __name__ == "__main__":
    main()
