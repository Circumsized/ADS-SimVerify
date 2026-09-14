#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "contracts/phase4"))
sys.path.insert(0, str(ROOT / "contracts/phase5"))
sys.path.insert(0, str(ROOT / "contracts/phase6"))

import phase5g_controlled_takeover as phase5g  # noqa: E402
import phase5h_articulation_takeover as phase5h  # noqa: E402
from oracle_nmpc_closed_loop import OracleGrid, Scenario, load_solver  # noqa: E402
from phase5b_shadow_replay import control_roi  # noqa: E402
from phase5c3_candidate_shadow import WarehouseCandidate  # noqa: E402
from phase6_matrix import aggregate_matrix, build_matrix  # noqa: E402


CONTRACT = ROOT / "contracts/phase6/phase6_contract.json"
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
PHASE5A = ROOT / "contracts/phase5/phase5a_status.json"
PHASE5K = ROOT / "contracts/phase5/phase5k_status.json"
ROBOT_PATH = "/Phase6Jetbot"
CAMERA_PATH = f"{ROBOT_PATH}/chassis/rgb_camera/jetbot_camera"
DYNAMIC_PATH = "/Phase6DynamicCart"


def perturb_rgb(image, rng, config):
    result = image.astype(np.float32) / 255.0
    result = np.power(np.clip(result, 0.0, 1.0), rng.uniform(*config["gamma"]))
    result = result * rng.uniform(*config["rgb_gain"])
    result += rng.uniform(*config["rgb_offset"]) / 255.0
    sigma = rng.uniform(*config["gaussian_noise_sigma_u8"]) / 255.0
    if sigma > 0.0:
        result += rng.normal(0.0, sigma, result.shape).astype(np.float32)
    result = np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)
    if rng.random() < config["motion_blur_probability"]:
        result = cv2.filter2D(result, -1, np.ones((1, 3), dtype=np.float32) / 3.0)
    quality = int(rng.integers(config["jpeg_quality"][0], config["jpeg_quality"][1] + 1))
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Phase 6 JPEG perturbation failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def perturb_depth(depth, rng, config):
    result = depth.astype(np.float32, copy=True)
    finite = np.isfinite(result)
    result[finite] *= rng.uniform(*config["scale"])
    sigma = rng.uniform(*config["noise_sigma_m"])
    if sigma > 0.0:
        result[finite] += rng.normal(0.0, sigma, int(finite.sum())).astype(np.float32)
    dropout = rng.uniform(*config["dropout_probability"])
    if dropout > 0.0:
        result[rng.random(result.shape) < dropout] = np.inf
    return result


def perturbed_scenario(base, cell):
    delta = cell["perturbations"]["start_goal"]
    return Scenario(
        base.name,
        (
            base.start[0] + delta["start_dx_m"],
            base.start[1] + delta["start_dy_m"],
            base.start[2] + delta["start_dyaw_rad"],
            base.start[3],
        ),
        (
            base.goal[0] + delta["goal_dx_m"],
            base.goal[1] + delta["goal_dy_m"],
            base.goal[2] + delta["goal_dyaw_rad"],
        ),
    )


def wheel_command_delta_p95(path):
    with Path(path).open(newline="", encoding="ascii") as source:
        values = [float(row["omega_radps"]) for row in csv.DictReader(source)]
    if len(values) < 2:
        return 0.0
    return float(np.percentile(np.abs(np.diff(np.asarray(values, dtype=np.float64))), 95.0))


class PerturbedModel:
    def __init__(self, model, rng, config):
        self.model = model
        self.rng = rng
        self.config = config

    def infer(self, image):
        return self.model.infer(perturb_rgb(image, self.rng, self.config))


class PairedDepthPerturber:
    def __init__(self, implementation, rng, config):
        self.implementation = implementation
        self.rng = rng
        self.config = config
        self.pending = None

    def __call__(self, classes, depth, sensor, bev):
        if self.pending is None:
            perturbed = perturb_depth(depth, self.rng, self.config)
            self.pending = perturbed
        else:
            perturbed = self.pending
            self.pending = None
        return self.implementation(classes, perturbed, sensor, bev)


def sensor_for_cell(base, cell):
    sensor = phase5h.scaled_sensor_geometry(base)
    delta = cell["perturbations"]["camera_extrinsics"]
    extrinsics = sensor["body_extrinsics"]
    extrinsics["height_m"] += delta["height_m"]
    extrinsics["yaw_rad"] += math.radians(delta["yaw_deg"])
    extrinsics["pitch_rad"] += math.radians(delta["pitch_deg"])
    return sensor


def dynamic_pose_for_cell(original, cell):
    config = cell["perturbations"]["dynamic_obstacle"]

    def pose(scenario_name, time_s):
        x_m, y_m = original(scenario_name, time_s + config["time_shift_s"])
        if x_m > 900.0:
            return x_m, y_m
        return x_m + config["forward_jitter_m"], y_m + config["lateral_jitter_m"]

    return pose


def inherited_phase5k_metrics(status):
    return {
        "hour_frames": int(status["hour_endurance"]["frames"]),
        "hour_duration_s": float(status["hour_endurance"]["wall_duration_s"]),
        "hour_collision_count": int(status["aggregate"]["collision_count"]),
        "maximum_fault_recovery_frames": int(
            status["fault_evidence"]["maximum_recoverable_fault_stop_latency_frames"]
        ),
    }


def finalize_matrix(output, evidence, summaries, inherited, contract, formal):
    columns = len(contract["matrix"]["scenarios"])
    image_rows = [
        np.concatenate(evidence[index:index + columns], axis=1)
        for index in range(0, len(evidence), columns)
    ]
    evidence_path = output / "evidence.png"
    if not cv2.imwrite(str(evidence_path), np.concatenate(image_rows, axis=0)):
        raise RuntimeError("failed to write Phase 6 matrix evidence")
    aggregate = aggregate_matrix(summaries, inherited, contract["acceptance"])
    passed = bool(formal and aggregate["gate_passed"])
    result = {
        "schema_version": "phase6-final-simulation-matrix-v1",
        "status": "phase6_gate_passed" if passed else ("phase6_gate_rejected" if formal else "smoke_only"),
        "cases": summaries,
        "aggregate": aggregate,
        "inherited_phase5k": inherited,
        "evidence": evidence_path.name,
        "evidence_sha256": phase5h.sha256(evidence_path),
        "real_vehicle_control_allowed": False,
        "gate_passed": passed,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    phase3 = json.loads(PHASE3.read_text())
    phase5a = json.loads(PHASE5A.read_text())
    phase5k = json.loads(PHASE5K.read_text())
    cells = build_matrix(contract)
    if args.case:
        cells = [cell for cell in cells if cell["case_id"] == args.case]
        if not cells:
            raise SystemExit(f"unknown Phase 6 case: {args.case}")
    output = args.output or ROOT / "artifacts/phase6_matrix" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    model_manifest = json.loads(phase5h.MODEL_MANIFEST.read_text())
    model_path = ROOT / model_manifest["model"]
    if phase5h.sha256(model_path) != model_manifest["model_sha256"]:
        raise SystemExit("warehouse candidate hash differs from manifest")
    oracle_manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    oracle_manifest = json.loads(oracle_manifest_path.read_text())
    grid = OracleGrid(oracle_manifest_path.parent / oracle_manifest["archive"], oracle_manifest)
    base_scenarios = {scenario.name: scenario for scenario in phase5g.scenarios()}
    bev = phase3["bev_contract"]
    fixed_roi = control_roi(bev)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "width": 320, "height": 240})
    summaries = []
    evidence = []
    try:
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.prims import Articulation, XFormPrim
        from isaacsim.core.utils.stage import open_stage
        from pxr import PhysxSchema, UsdGeom, UsdLux, UsdPhysics

        open_stage(str(phase5h.PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        robot_prim = stage.DefinePrim(ROBOT_PATH, "Xform")
        robot_prim.GetReferences().AddReference(str(phase5h.JETBOT_SOURCE), "/Root/jetbot")
        for prim in stage.Traverse():
            if prim.GetPath().HasPrefix(ROBOT_PATH) and prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(1e5)
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr().Set(100000.0)
        obstacle = UsdGeom.Cube.Define(stage, DYNAMIC_PATH)
        obstacle.CreateSizeAttr(phase5h.DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle_color = obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        rigid = UsdPhysics.RigidBodyAPI.Apply(obstacle.GetPrim())
        rigid.CreateKinematicEnabledAttr().Set(True)
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
        lights = []
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.RectLight):
                attr = UsdLux.RectLight(prim).GetIntensityAttr()
                value = attr.Get()
                if value is not None:
                    lights.append((attr, float(value)))
        camera = UsdGeom.Camera(stage.GetPrimAtPath(CAMERA_PATH))
        source_sensor = phase5h.scaled_sensor_geometry(phase3["sensor_geometry"])
        width, height = source_sensor["image_size"]
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(
            source_sensor["intrinsics"]["fx"] * aperture * height
            / (source_sensor["intrinsics"]["fy"] * width)
        )
        camera.CreateFocalLengthAttr(source_sensor["intrinsics"]["fx"] * aperture / width)
        product = rep.create.render_product(CAMERA_PATH, (width, height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb.attach([product])
        depth.attach([product])
        world = World(physics_dt=phase5h.PHYSICS_DT, rendering_dt=phase5h.PHYSICS_DT, backend="numpy")
        car = Articulation(prim_paths_expr=ROBOT_PATH, name="phase6_jetbot")
        dynamic = XFormPrim(prim_paths_expr=DYNAMIC_PATH, name="phase6_dynamic_cart")
        world.scene.add(car)
        world.scene.add(dynamic)
        world.reset()
        if car.dof_names != ["left_wheel_joint", "right_wheel_joint"]:
            raise RuntimeError(f"unexpected wheel DOFs: {car.dof_names}")
        world.play()
        base_model = WarehouseCandidate(model_path)
        for _ in range(3):
            base_model.infer(np.zeros((240, 320, 3), dtype=np.uint8))
        phase5h.reset_robot(car, world, base_scenarios[cells[0]["scenario"]], dynamic)
        for _ in range(10):
            world.step(render=True)
            rgb.get_data()
            depth.get_data()

        original_depth_lift = phase5h.candidate_depth_lift
        original_h_dynamic = phase5h.dynamic_pose
        original_g_dynamic = phase5g.dynamic_pose
        for index, cell in enumerate(cells):
            case_output = output / cell["case_id"]
            case_output.mkdir()
            for attr, baseline in lights:
                attr.Set(baseline * cell["perturbations"]["lighting"]["intensity_scale"])
            obstacle_color.Set([tuple(cell["perturbations"]["material"]["dynamic_obstacle_rgb"])])
            scenario = perturbed_scenario(base_scenarios[cell["scenario"]], cell)
            sensor = sensor_for_cell(phase3["sensor_geometry"], cell)
            seed = int(cell["seed"]) + index * 1009
            model = PerturbedModel(
                base_model, np.random.default_rng(seed), cell["perturbations"]["jpeg_rgb"]
            )
            depth_lift = PairedDepthPerturber(
                original_depth_lift,
                np.random.default_rng(seed + 1),
                cell["perturbations"]["metric_depth"],
            )
            dynamic_pose = dynamic_pose_for_cell(original_g_dynamic, cell)
            phase5h.candidate_depth_lift = depth_lift
            phase5h.dynamic_pose = dynamic_pose
            phase5g.dynamic_pose = dynamic_pose
            try:
                summary, image = phase5h.run_scenario(
                    scenario, grid, load_solver(), model, sensor, bev, fixed_roi,
                    world, car, dynamic, rgb, depth, case_output, args.max_steps,
                )
            finally:
                phase5h.candidate_depth_lift = original_depth_lift
                phase5h.dynamic_pose = original_h_dynamic
                phase5g.dynamic_pose = original_g_dynamic
            summary.update({
                "case_id": cell["case_id"],
                "seed": cell["seed"],
                "scenario": cell["scenario"],
                "profile": cell["profile"],
                "perturbations": cell["perturbations"],
                "telemetry": f"{cell['case_id']}/{summary['telemetry']}",
                "wheel_command_delta_p95": wheel_command_delta_p95(
                    case_output / f"{scenario.name}.csv"
                ),
            })
            summaries.append(summary)
            cv2.putText(image, cell["case_id"], (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            evidence.append(image)
            print(
                f"[Phase 6] {cell['case_id']} reached={summary['reached']} "
                f"abort={summary['abort_reason']} p95={summary['sensor_to_wheel_p95_ms']:.2f}ms"
            )
        rgb.detach([product])
        depth.detach([product])
        world.stop()
        inherited = inherited_phase5k_metrics(phase5k)
        formal = len(cells) == contract["acceptance"]["required_cases"] and args.max_steps is None
        result = finalize_matrix(output, evidence, summaries, inherited, contract, formal)
        print(json.dumps(result["aggregate"], indent=2))
        print(f"Phase 6 artifacts: {output}")
        if formal and not result["gate_passed"]:
            raise SystemExit(1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
