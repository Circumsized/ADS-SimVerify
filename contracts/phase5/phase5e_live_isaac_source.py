#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contracts/phase4"))
sys.path.insert(0, str(ROOT / "contracts/phase5"))
from capture_smoke_dataset import camera_matrix  # noqa: E402
from phase5b_shadow_replay import (  # noqa: E402
    control_roi,
    load_trajectory,
    nearest_center_obstacle,
    oracle_bev,
)
from phase5c2_geometry_upper_bound import dynamic_footprint_mask  # noqa: E402
from phase5c3_candidate_shadow import candidate_depth_lift  # noqa: E402
from phase5c_dynamic_upper_bound import (  # noqa: E402
    DYNAMIC_HALF_EXTENT_M,
    STOP_DISTANCE_M,
    dynamic_case,
    world_from_ego,
)


def main():
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    phase5c2 = json.loads((ROOT / "contracts/phase5/phase5c2_status.json").read_text())
    contract = json.loads((ROOT / "contracts/phase5/phase5e_contract.json").read_text())
    frame_count = int(
        os.environ.get("PHASE5E_MAX_FRAMES", contract["live_isaac_shadow"]["frames"])
    )
    _, trajectory = load_trajectory(phase5a)
    trajectory = trajectory[:frame_count]
    oracle_manifest_path = ROOT / phase5c2["perception_oracle"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_manifest_path.read_text())
    oracle_archive = np.load(ROOT / phase5c2["perception_oracle"]["archive"]["path"])
    perception_occupied = oracle_archive["perception_occupied"]
    sensor, bev = phase3["sensor_geometry"], phase3["bev_contract"]
    fixed_roi = control_roi(bev)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom

        open_stage(str(ROOT / "assets/phase4/warehouse_nav14_overlay.usda"))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        obstacle = UsdGeom.Cube.Define(stage, "/Phase5ELiveDynamicObstacle")
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        obstacle_translate = UsdGeom.Xformable(obstacle).AddTranslateOp()
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
        camera = UsdGeom.Camera.Define(stage, "/Phase5ELiveCamera")
        width, height = sensor["image_size"]
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(
            sensor["intrinsics"]["fx"] * aperture * height
            / (sensor["intrinsics"]["fy"] * width)
        )
        camera.CreateFocalLengthAttr(sensor["intrinsics"]["fx"] * aperture / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        camera_transform = UsdGeom.Xformable(camera).AddTransformOp()
        product = rep.create.render_product(str(camera.GetPath()), (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb_annotator.attach([product])
        depth_annotator.attach([product])

        node = Node()
        frame_id = 0
        while frame_id < len(trajectory):
            event = node.next(timeout=1.0)
            if event is None:
                continue
            if event["type"] == "STOP":
                break
            if event["type"] != "INPUT" or event["id"] != "tick":
                continue
            frame = trajectory[frame_id]
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
            camera_transform.Set(camera_matrix(eye, yaw + ext["yaw_rad"], ext["pitch_rad"], Gf))
            rep.orchestrator.step()
            rgb = np.asarray(rgb_annotator.get_data())[:, :, :3].astype(np.uint8)
            depth = np.asarray(depth_annotator.get_data(), dtype=np.float32)
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                raise RuntimeError("Phase 5-E live JPEG encoding failed")
            dummy_classes = np.full(depth.shape, 13, dtype=np.uint8)
            _, depth_reference_valid = candidate_depth_lift(dummy_classes, depth, sensor, bev)
            oracle, oracle_valid = oracle_bev(perception_occupied, oracle_manifest, (x, y, yaw), bev)
            if dynamic_forward > 0.0:
                oracle |= dynamic_footprint_mask(bev, dynamic_forward, dynamic_left)
            nearest = nearest_center_obstacle(oracle, fixed_roi & oracle_valid, bev)
            metadata = {
                "source_frame_id": frame_id,
                "source_kind": "phase5e_live_isaac",
                "dynamic_mode": mode,
                "oracle_stop": bool(nearest is not None and nearest <= STOP_DISTANCE_M),
                "depth_units": "meters",
                "depth_measurement": "distance_to_image_plane",
            }
            node.send_output("jpeg_image", pa.array(encoded.ravel(), type=pa.uint8()), metadata=metadata)
            node.send_output("metric_depth", pa.array(depth.ravel(), type=pa.float32()), metadata=metadata)
            node.send_output("oracle_bev", pa.array(np.where(oracle, 255, 0).ravel(), type=pa.uint8()), metadata=metadata)
            node.send_output("oracle_valid", pa.array(oracle_valid.astype(np.uint8).ravel(), type=pa.uint8()), metadata=metadata)
            node.send_output("depth_reference_valid", pa.array(depth_reference_valid.astype(np.uint8).ravel(), type=pa.uint8()), metadata=metadata)
            frame_id += 1
            if frame_id % 100 == 0:
                print(f"[Phase 5-E live source] frames={frame_id}/{len(trajectory)} mode={mode}")
        print(f"[Phase 5-E live source] emitted {frame_id} live synchronized frames")
    except BaseException:
        app.close()
        raise
    app.close()


if __name__ == "__main__":
    main()
