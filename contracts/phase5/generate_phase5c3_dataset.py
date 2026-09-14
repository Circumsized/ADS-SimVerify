#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/phase5/phase5c3_contract.json"
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
PHASE5A_STATUS = ROOT / "contracts/phase5/phase5a_status.json"
PHASE5C2_STATUS = ROOT / "contracts/phase5/phase5c2_status.json"
PHASE4_DIR = ROOT / "contracts/phase4"
PHASE5_DIR = ROOT / "contracts/phase5"
sys.path.insert(0, str(PHASE4_DIR))
sys.path.insert(0, str(PHASE5_DIR))
from capture_smoke_dataset import camera_matrix, semantic_id_image  # noqa: E402
from phase5b_shadow_replay import load_trajectory  # noqa: E402
from warehouse_semantics import CHANNELS  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_from_ego(x, y, yaw, forward, left):
    return (
        x + forward * math.cos(yaw) - left * math.sin(yaw),
        y + forward * math.sin(yaw) + left * math.cos(yaw),
    )


def load_pose_pools(contract, phase5a, oracle_manifest, occupied, np, cv2):
    _, gate_trajectory = load_trajectory(phase5a)
    gate_xy = np.asarray(
        [(frame["x_m"], frame["y_m"]) for frame in gate_trajectory], dtype=np.float32
    )
    free = (occupied == 0).astype(np.uint8)
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * oracle_manifest["resolution_m"]
    min_x, max_x, min_y, max_y = oracle_manifest["bounds_xy_m"]
    resolution = oracle_manifest["resolution_m"]
    exclusion = contract["dataset"]["phase5_gate_trajectory_exclusion_m"]
    pools = {"train": [], "validation": []}
    for row, col in zip(*np.nonzero(clearance >= 0.75)):
        x = min_x + col * resolution
        y = max_y - row * resolution
        if not (min_x + 0.5 <= x <= max_x - 0.5 and min_y + 0.5 <= y <= max_y - 0.5):
            continue
        if float(np.min(np.linalg.norm(gate_xy - (x, y), axis=1))) < exclusion:
            continue
        block_x = math.floor((x - min_x) / 2.0)
        block_y = math.floor((y - min_y) / 2.0)
        split = "validation" if (block_x * 17 + block_y * 31) % 5 == 0 else "train"
        pools[split].append((x, y))
    if min(len(pools["train"]), len(pools["validation"])) < 50:
        raise RuntimeError(f"insufficient spatially isolated camera poses: {pools}")
    return pools


def create_dynamic_primitives(stage, add_labels, UsdGeom):
    specs = []

    curb = UsdGeom.Cube.Define(stage, "/Phase5C3Dynamic/Curb")
    curb.CreateSizeAttr(1.0)
    curb.CreateDisplayColorAttr([(0.85, 0.75, 0.10)])
    curb_xform = UsdGeom.Xformable(curb)
    curb_translate = curb_xform.AddTranslateOp()
    curb_xform.AddScaleOp().Set((0.55, 0.12, 0.08))
    add_labels(curb.GetPrim(), labels=["curb_or_step"], taxonomy="class")
    specs.append(("curb_or_step", curb_translate, 0.08))

    person = UsdGeom.Cylinder.Define(stage, "/Phase5C3Dynamic/Person")
    person.CreateRadiusAttr(0.15)
    person.CreateHeightAttr(1.60)
    person.CreateDisplayColorAttr([(0.15, 0.35, 0.90)])
    person_translate = UsdGeom.Xformable(person).AddTranslateOp()
    add_labels(person.GetPrim(), labels=["person"], taxonomy="class")
    specs.append(("person", person_translate, 0.80))

    cart = UsdGeom.Cube.Define(stage, "/Phase5C3Dynamic/Cart")
    cart.CreateSizeAttr(1.0)
    cart.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
    cart_xform = UsdGeom.Xformable(cart)
    cart_translate = cart_xform.AddTranslateOp()
    cart_xform.AddScaleOp().Set((0.45, 0.35, 0.45))
    add_labels(cart.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
    specs.append(("robot_or_cart", cart_translate, 0.45))

    forklift = UsdGeom.Cube.Define(stage, "/Phase5C3Dynamic/Forklift")
    forklift.CreateSizeAttr(1.0)
    forklift.CreateDisplayColorAttr([(0.95, 0.45, 0.05)])
    forklift_xform = UsdGeom.Xformable(forklift)
    forklift_translate = forklift_xform.AddTranslateOp()
    forklift_xform.AddScaleOp().Set((0.90, 0.55, 0.90))
    add_labels(
        forklift.GetPrim(), labels=["forklift_or_heavy_vehicle"], taxonomy="class"
    )
    specs.append(("forklift_or_heavy_vehicle", forklift_translate, 0.90))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frames", type=int)
    parser.add_argument("--validation-frames", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    train_frames = args.train_frames or contract["dataset"]["train_frames"]
    validation_frames = args.validation_frames or contract["dataset"]["validation_frames"]
    if train_frames < 1 or validation_frames < 1:
        raise SystemExit("dataset split sizes must be positive")
    output = args.output or ROOT / "artifacts/phase5c3_dataset" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "validation"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    phase5a = json.loads(PHASE5A_STATUS.read_text(encoding="utf-8"))
    phase5c2 = json.loads(PHASE5C2_STATUS.read_text(encoding="utf-8"))
    oracle_manifest_path = ROOT / phase5c2["perception_oracle"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_manifest_path.read_text(encoding="utf-8"))
    oracle_archive = ROOT / phase5c2["perception_oracle"]["archive"]["path"]
    sensor = phase3["sensor_geometry"]
    seed = contract["dataset"]["seed"]

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    split_counts = {"train": Counter(), "validation": Counter()}
    try:
        import cv2
        import numpy as np
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom

        occupied = np.load(oracle_archive)["perception_occupied"]
        pools = load_pose_pools(contract, phase5a, oracle_manifest, occupied, np, cv2)
        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        dynamic = create_dynamic_primitives(stage, add_labels, UsdGeom)

        camera = UsdGeom.Camera.Define(stage, "/Phase5C3DatasetCamera")
        width, height = contract["dataset"]["image_size"]
        horizontal_aperture = 20.955
        scaled_fx = sensor["intrinsics"]["fx"] * width / sensor["image_size"][0]
        scaled_fy = sensor["intrinsics"]["fy"] * height / sensor["image_size"][1]
        vertical_aperture = scaled_fx * horizontal_aperture * height / (scaled_fy * width)
        focal_length = scaled_fx * horizontal_aperture / width
        camera.CreateHorizontalApertureAttr(horizontal_aperture)
        camera.CreateVerticalApertureAttr(vertical_aperture)
        camera.CreateFocalLengthAttr(focal_length)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera_transform = UsdGeom.Xformable(camera).AddTransformOp()
        render_product = rep.create.render_product(str(camera.GetPath()), (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False}
        )
        rgb_annotator.attach([render_product])
        semantic_annotator.attach([render_product])

        manifest_csv = output / "frames.csv"
        with manifest_csv.open("w", newline="", encoding="ascii") as target:
            fields = ["split", "frame_id", "x_m", "y_m", "yaw_rad", "image", "label"]
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            global_frame = 0
            for split, count in (("train", train_frames), ("validation", validation_frames)):
                rng = random.Random(seed + (0 if split == "train" else 1))
                for frame_id in range(count):
                    x, y = pools[split][rng.randrange(len(pools[split]))]
                    yaw = rng.uniform(-math.pi, math.pi)
                    ext = sensor["body_extrinsics"]
                    eye = (
                        x + math.cos(yaw) * ext["forward_m"] - math.sin(yaw) * ext["left_m"],
                        y + math.sin(yaw) * ext["forward_m"] + math.cos(yaw) * ext["left_m"],
                        ext["height_m"],
                    )
                    camera_transform.Set(
                        camera_matrix(eye, yaw + ext["yaw_rad"], ext["pitch_rad"], Gf)
                    )
                    lateral_slots = [-1.45, -0.48, 0.48, 1.45]
                    rng.shuffle(lateral_slots)
                    for index, (_, translate, z) in enumerate(dynamic):
                        forward = 1.0 + index * 0.85 + rng.uniform(-0.20, 0.20)
                        left = lateral_slots[index] + rng.uniform(-0.12, 0.12)
                        obj_x, obj_y = world_from_ego(x, y, yaw, forward, left)
                        translate.Set(Gf.Vec3d(obj_x, obj_y, z))
                    rep.orchestrator.step()
                    rgb = np.asarray(rgb_annotator.get_data())[:, :, :3].astype(np.uint8)
                    labels, _ = semantic_id_image(semantic_annotator.get_data(), np)
                    image_rel = Path("images") / split / f"{frame_id:06d}.jpg"
                    label_rel = Path("labels") / split / f"{frame_id:06d}.png"
                    if not cv2.imwrite(
                        str(output / image_rel),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92],
                    ):
                        raise RuntimeError("failed to write training image")
                    if not cv2.imwrite(str(output / label_rel), labels):
                        raise RuntimeError("failed to write semantic label")
                    counts = np.bincount(labels.ravel(), minlength=len(CHANNELS))
                    split_counts[split].update(
                        {CHANNELS[index]: int(value) for index, value in enumerate(counts)}
                    )
                    row = {
                        "split": split,
                        "frame_id": frame_id,
                        "x_m": x,
                        "y_m": y,
                        "yaw_rad": yaw,
                        "image": str(image_rel),
                        "label": str(label_rel),
                    }
                    writer.writerow(row)
                    global_frame += 1
                    if global_frame % 100 == 0:
                        print(
                            f"[Phase 5-C3 dataset] frames={global_frame}/{train_frames + validation_frames} split={split}"
                        )
    except BaseException:
        app.close()
        raise

    missing = {
        split: [name for name in CHANNELS if split_counts[split][name] == 0]
        for split in ("train", "validation")
    }
    summary = {
        "schema_version": "phase5c3-dataset-v1",
        "status": "complete" if not any(missing.values()) else "incomplete_class_coverage",
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_sha256": sha256(CONTRACT),
        "source_overlay": str(PHASE4_OVERLAY.relative_to(ROOT)),
        "source_overlay_sha256": sha256(PHASE4_OVERLAY),
        "perception_oracle_manifest": str(oracle_manifest_path.relative_to(ROOT)),
        "perception_oracle_manifest_sha256": sha256(oracle_manifest_path),
        "seed": seed,
        "image_size": [width, height],
        "splits": {
            "train": {"frames": train_frames, "class_pixels": dict(split_counts["train"])},
            "validation": {
                "frames": validation_frames,
                "class_pixels": dict(split_counts["validation"]),
            },
        },
        "missing_classes": missing,
        "frame_manifest": "frames.csv",
        "frame_manifest_sha256": sha256(output / "frames.csv"),
        "phase5_gate_frames_used": False,
        "phase4_smoke_frames_used": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2))
    print(f"Phase 5-C3 dataset: {output}")
    app.close()


if __name__ == "__main__":
    main()
