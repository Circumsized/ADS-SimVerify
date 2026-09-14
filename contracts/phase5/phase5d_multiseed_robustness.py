#!/usr/bin/env python3
import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from phase5b_shadow_replay import (
    aggregate,
    control_roi,
    load_trajectory,
    nearest_center_obstacle,
    occupancy_from_semantic,
    occupancy_panel,
    oracle_bev,
)
from phase5c2_geometry_upper_bound import dynamic_footprint_mask, occupancy_metrics_contract
from phase5c3_candidate_shadow import WarehouseCandidate, candidate_depth_lift
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
CONTRACT = ROOT / "contracts/phase5/phase5d_contract.json"
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
PHASE5A_STATUS = ROOT / "contracts/phase5/phase5a_status.json"
PHASE5C2_STATUS = ROOT / "contracts/phase5/phase5c2_status.json"
MODEL_MANIFEST = ROOT / "model/warehouse_nav14_candidate.json"
FIELDS = [
    "source_frame_id",
    "dynamic_mode",
    "oracle_stop",
    "candidate_stop",
    "candidate_valid_ratio",
    "candidate_latency_ms",
    "occupied_iou",
    "free_iou",
    "false_free_rate",
    "false_occupied_rate",
]


def perturbed_rgb(rgb, rng, limits):
    image = rgb.astype(np.float32) / 255.0
    gamma = rng.uniform(*limits["gamma"])
    gain = rng.uniform(*limits["rgb_gain"])
    offset = rng.uniform(*limits["rgb_offset"]) / 255.0
    image = np.power(np.clip(image, 0.0, 1.0), gamma) * gain + offset
    sigma = rng.uniform(*limits["gaussian_noise_sigma_u8"]) / 255.0
    if sigma > 0.0:
        image += rng.normal(0.0, sigma, image.shape).astype(np.float32)
    image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if rng.random() < limits["motion_blur_probability"]:
        image = cv2.filter2D(image, -1, np.ones((1, 3), dtype=np.float32) / 3.0)
    quality = int(rng.integers(limits["jpeg_quality"][0], limits["jpeg_quality"][1] + 1))
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("Phase 5-D JPEG perturbation failed")
    return encoded, cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def perturbed_depth(depth, rng, limits):
    result = depth.astype(np.float32, copy=True)
    finite = np.isfinite(result)
    scale = rng.uniform(*limits["depth_scale"])
    sigma = rng.uniform(*limits["depth_noise_sigma_m"])
    result[finite] *= scale
    if sigma > 0.0:
        result[finite] += rng.normal(0.0, sigma, int(finite.sum())).astype(np.float32)
    dropout = rng.uniform(*limits["depth_dropout_probability"])
    if dropout > 0.0:
        result[rng.random(result.shape) < dropout] = np.inf
    return result


def camera_pose(sensor, profile, x, y, yaw):
    delta = profile["camera_delta"]
    ext = sensor["body_extrinsics"]
    height = ext["height_m"] + delta["height_m"]
    camera_yaw = ext["yaw_rad"] + math.radians(delta["yaw_deg"])
    camera_pitch = ext["pitch_rad"] + math.radians(delta["pitch_deg"])
    eye = (
        x + math.cos(yaw) * ext["forward_m"] - math.sin(yaw) * ext["left_m"],
        y + math.sin(yaw) * ext["forward_m"] + math.cos(yaw) * ext["left_m"],
        height,
    )
    return eye, yaw + camera_yaw, camera_pitch


def calibrated_sensor(sensor, profile):
    result = json.loads(json.dumps(sensor))
    delta = profile["camera_delta"]
    ext = result["body_extrinsics"]
    ext["height_m"] += delta["height_m"]
    ext["yaw_rad"] += math.radians(delta["yaw_deg"])
    ext["pitch_rad"] += math.radians(delta["pitch_deg"])
    return result


def method_passed(rows, acceptance):
    metrics = {name: aggregate(rows, name) for name in (
        "occupied_iou", "free_iou", "false_free_rate", "false_occupied_rate"
    )}
    stops = stop_metrics(
        [{"oracle_stop": bool(row["oracle_stop"]), "depth_gt_stop": bool(row["candidate_stop"])} for row in rows]
    )
    valid = aggregate(rows, "candidate_valid_ratio")
    latency = aggregate(rows, "candidate_latency_ms")
    passed = bool(
        metrics["occupied_iou"]["mean"] >= acceptance["occupied_iou_mean_min"]
        and metrics["free_iou"]["mean"] >= acceptance["free_iou_mean_min"]
        and metrics["false_free_rate"]["mean"] <= acceptance["false_free_rate_mean_max"]
        and metrics["false_occupied_rate"]["mean"] <= acceptance["false_occupied_rate_mean_max"]
        and stops["stop_recall"] >= acceptance["stop_recall_min"]
        and stops["go_specificity"] >= acceptance["go_specificity_min"]
        and valid["mean"] >= acceptance["candidate_valid_ratio_mean_min"]
        and latency["p95"] <= acceptance["latency_p95_ms_max"]
    )
    return {"metrics": metrics, "stop_decision": stops, "valid_ratio": valid, "latency_ms": latency, "passed": passed}


def write_evidence(path, decoded, oracle, candidate, valid, title):
    rgb = cv2.resize(decoded, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(rgb, (0, 0), (319, 28), (15, 15, 15), -1)
    cv2.putText(rgb, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    error = np.zeros((*oracle.shape, 3), dtype=np.uint8)
    error[valid & (oracle == candidate)] = (55, 120, 55)
    error[valid & oracle & (~candidate)] = (0, 0, 255)
    error[valid & (~oracle) & candidate] = (0, 190, 255)
    error = cv2.resize(error, (320, 240), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack((rgb, occupancy_panel(oracle, valid, "Oracle"), occupancy_panel(candidate, valid, "Perturbed candidate"), error))
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write evidence: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--profiles", type=int)
    parser.add_argument("--fixture-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    frame_count = args.max_frames or contract["robustness"]["frames_per_seed"]
    profiles = contract["robustness"]["profiles"][: args.profiles]
    if not profiles or not 1 <= frame_count <= FRAME_COUNT:
        raise SystemExit("invalid profile/frame selection")
    fixture_frames = args.fixture_frames
    if fixture_frames is None:
        fixture_frames = min(contract["runtime"]["fixture_frames"], frame_count)
    output = args.output or ROOT / "artifacts/phase5d_robustness" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    evidence_dir = output / "evidence"
    fixture_dir = output / "runtime_fixture"
    evidence_dir.mkdir()
    fixture_dir.mkdir()
    (fixture_dir / "jpeg").mkdir()
    (fixture_dir / "depth").mkdir()
    (fixture_dir / "expected").mkdir()

    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    phase5a = json.loads(PHASE5A_STATUS.read_text(encoding="utf-8"))
    phase5c2 = json.loads(PHASE5C2_STATUS.read_text(encoding="utf-8"))
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    model_path = ROOT / model_manifest["model"]
    if sha256(model_path) != model_manifest["model_sha256"]:
        raise SystemExit("warehouse candidate hash differs from manifest")
    _, trajectory = load_trajectory(phase5a)
    trajectory = trajectory[:frame_count]
    oracle_manifest_path = ROOT / phase5c2["perception_oracle"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    oracle_archive = np.load(ROOT / phase5c2["perception_oracle"]["archive"]["path"])
    perception_occupied = oracle_archive["perception_occupied"]
    sensor, bev = phase3["sensor_geometry"], phase3["bev_contract"]
    fixed_roi = control_roi(bev)
    limits = contract["robustness"]["per_frame_perturbation"]
    model = WarehouseCandidate(model_path)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    profile_results = []
    fixture_manifest = []
    try:
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom
        from capture_smoke_dataset import camera_matrix

        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        obstacle = UsdGeom.Cube.Define(stage, "/Phase5DRobustnessDynamicObstacle")
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        obstacle_translate = UsdGeom.Xformable(obstacle).AddTranslateOp()
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
        camera = UsdGeom.Camera.Define(stage, "/Phase5DRobustnessCamera")
        width, height = sensor["image_size"]
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(sensor["intrinsics"]["fx"] * aperture * height / (sensor["intrinsics"]["fy"] * width))
        camera.CreateFocalLengthAttr(sensor["intrinsics"]["fx"] * aperture / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera_transform = UsdGeom.Xformable(camera).AddTransformOp()
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb_annotator.attach([product])
        depth_annotator.attach([product])

        for profile_index, profile in enumerate(profiles):
            rows = []
            profile_sensor = calibrated_sensor(sensor, profile)
            csv_path = output / f"seed_{profile['seed']}.csv"
            with csv_path.open("w", newline="", encoding="ascii") as target:
                writer = csv.DictWriter(target, fieldnames=FIELDS)
                writer.writeheader()
                for frame_id, frame in enumerate(trajectory):
                    rng = np.random.default_rng(profile["seed"] + frame_id * 104729)
                    x, y, yaw = frame["x_m"], frame["y_m"], frame["yaw_rad"]
                    mode, dynamic_forward, dynamic_left = dynamic_case(frame_id)
                    if dynamic_forward > 0.0:
                        dynamic_forward += rng.uniform(*limits["dynamic_forward_jitter_m"])
                        dynamic_left += rng.uniform(*limits["dynamic_lateral_jitter_m"])
                    object_x, object_y = world_from_ego(x, y, yaw, dynamic_forward, dynamic_left)
                    obstacle_translate.Set(Gf.Vec3d(object_x, object_y, DYNAMIC_HALF_EXTENT_M))
                    eye, camera_yaw, camera_pitch = camera_pose(sensor, profile, x, y, yaw)
                    camera_transform.Set(camera_matrix(eye, camera_yaw, camera_pitch, Gf))
                    rep.orchestrator.step()
                    rgb = np.asarray(rgb_annotator.get_data())[:, :, :3].astype(np.uint8)
                    raw_depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
                    encoded, decoded = perturbed_rgb(rgb, rng, limits)
                    depth = perturbed_depth(raw_depth, rng, limits)
                    if profile_index == 0 and frame_id < fixture_frames:
                        depth = depth.astype(np.float16).astype(np.float32)
                    candidate_ids, inference_ms = model.infer(decoded)
                    started = time.perf_counter_ns()
                    candidate_bev, candidate_valid = candidate_depth_lift(candidate_ids, depth, profile_sensor, bev)
                    _, clean_depth_valid = candidate_depth_lift(
                        candidate_ids, raw_depth, profile_sensor, bev
                    )
                    total_ms = inference_ms + (time.perf_counter_ns() - started) / 1e6
                    candidate_occupied = occupancy_from_semantic(candidate_bev)
                    oracle, oracle_valid = oracle_bev(perception_occupied, oracle_manifest, (x, y, yaw), bev)
                    if dynamic_forward > 0.0:
                        oracle |= dynamic_footprint_mask(bev, dynamic_forward, dynamic_left)
                    clean_visible_roi = fixed_roi & oracle_valid & clean_depth_valid
                    common_valid = clean_visible_roi & candidate_valid
                    metrics = occupancy_metrics_contract(oracle, candidate_occupied, common_valid)
                    nearest_oracle = nearest_center_obstacle(oracle, fixed_roi & oracle_valid, bev)
                    nearest_candidate = nearest_center_obstacle(candidate_occupied, fixed_roi & candidate_valid, bev)
                    row = {
                        "source_frame_id": frame_id,
                        "dynamic_mode": mode,
                        "oracle_stop": int(nearest_oracle is not None and nearest_oracle <= STOP_DISTANCE_M),
                        "candidate_stop": int(nearest_candidate is not None and nearest_candidate <= STOP_DISTANCE_M),
                        "candidate_valid_ratio": float(
                            common_valid.sum() / max(clean_visible_roi.sum(), 1)
                        ),
                        "candidate_latency_ms": total_ms,
                        **{name: metrics[name] for name in ("occupied_iou", "free_iou", "false_free_rate", "false_occupied_rate")},
                    }
                    writer.writerow(row)
                    rows.append(row)
                    if profile_index == 0 and frame_id < fixture_frames:
                        jpeg_path = fixture_dir / "jpeg" / f"{frame_id:06d}.jpg"
                        depth_path = fixture_dir / "depth" / f"{frame_id:06d}.npy"
                        expected_path = fixture_dir / "expected" / f"{frame_id:06d}.npy"
                        jpeg_path.write_bytes(encoded.tobytes())
                        np.save(depth_path, depth.astype(np.float16))
                        np.save(expected_path, np.where(candidate_occupied, 255, 0).astype(np.uint8))
                        fixture_manifest.append({
                            "source_frame_id": frame_id,
                            "jpeg": str(jpeg_path.relative_to(fixture_dir)),
                            "depth": str(depth_path.relative_to(fixture_dir)),
                            "expected": str(expected_path.relative_to(fixture_dir)),
                        })
                    if frame_id in {0, frame_count // 2, frame_count - 1}:
                        evidence_path = evidence_dir / f"{profile['name']}_{frame_id:06d}.png"
                        write_evidence(evidence_path, decoded, oracle, candidate_occupied, common_valid, f"{profile['name']} {mode}")
                    if (frame_id + 1) % 100 == 0 or frame_id + 1 == frame_count:
                        print(f"[Phase 5-D] profile={profile['name']} frames={frame_id + 1}/{frame_count} latency={total_ms:.2f}ms")
            result = method_passed(rows, contract["robustness"]["acceptance"])
            result.update({"name": profile["name"], "seed": profile["seed"], "frames": len(rows), "telemetry": csv_path.name, "telemetry_sha256": sha256(csv_path)})
            profile_results.append(result)
    except BaseException:
        app.close()
        raise

    fixture_manifest_path = fixture_dir / "manifest.json"
    fixture_manifest_path.write_text(json.dumps({
        "schema_version": "phase5d-runtime-fixture-v1",
        "frames": fixture_manifest,
        "image_shape": [480, 640, 3],
        "depth_shape": [480, 640],
        "depth_storage": "float16 meters; source promotes to Float32",
        "control_authority": False,
    }, indent=2) + "\n", encoding="ascii")
    formal = frame_count == contract["robustness"]["frames_per_seed"] and len(profiles) == len(contract["robustness"]["profiles"])
    summary = {
        "schema_version": "phase5d-multiseed-v1",
        "status": "robustness_passed" if formal and all(item["passed"] for item in profile_results) else "smoke_or_rejected",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT),
        "candidate": str(model_path.relative_to(ROOT)),
        "candidate_sha256": sha256(model_path),
        "profiles": profile_results,
        "all_profiles_passed": bool(formal and all(item["passed"] for item in profile_results)),
        "runtime_fixture": str(fixture_manifest_path.relative_to(output)),
        "runtime_fixture_manifest_sha256": sha256(fixture_manifest_path),
        "control_promotion_allowed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps({"status": summary["status"], "profiles": [{"name": item["name"], "passed": item["passed"]} for item in profile_results]}, indent=2))
    print(f"Phase 5-D robustness artifacts: {output}")
    app.close()


if __name__ == "__main__":
    main()
