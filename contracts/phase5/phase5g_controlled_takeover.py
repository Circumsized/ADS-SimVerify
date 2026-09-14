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
sys.path.insert(0, str(ROOT / "contracts/phase4"))
sys.path.insert(0, str(ROOT / "contracts/phase5"))
from capture_smoke_dataset import camera_matrix  # noqa: E402
from oracle_nmpc_closed_loop import (  # noqa: E402
    DT,
    OracleGrid,
    Polyline,
    Scenario,
    astar,
    load_solver,
    prune_path,
    wrap_angle,
)
from phase5b_shadow_replay import control_roi, occupancy_from_semantic  # noqa: E402
from phase5c3_candidate_shadow import (  # noqa: E402
    WarehouseCandidate,
    candidate_depth_lift,
)
from phase5f_dual_nmpc_shadow import (  # noqa: E402
    bev_obstacle_parameters,
    solve_shadow,
)

PHASE4_OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
MODEL_MANIFEST = ROOT / "model/warehouse_nav14_candidate.json"
DYNAMIC_HALF_EXTENT_M = 0.225
CSV_FIELDS = [
    "step", "time_s", "x_m", "y_m", "yaw_rad", "velocity_mps",
    "acceleration_mps2", "omega_radps", "target_mode", "goal_distance_m",
    "goal_yaw_error_rad", "path_progress_m", "path_error_m",
    "candidate_valid_ratio", "candidate_obstacle_count", "candidate_minimum_h",
    "solver_status", "render_ms", "candidate_pipeline_ms", "sensor_to_command_ms",
    "dynamic_x_m", "dynamic_y_m", "dynamic_center_distance_m",
    "static_collision", "dynamic_collision", "supervisor_decision",
    "command_applied", "candidate_controls_simulation",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrate_state(state, acceleration, omega):
    result = np.asarray(state, dtype=np.float64).copy()
    acceleration = float(np.clip(acceleration, -1.0, 1.0))
    omega = float(np.clip(omega, -0.6, 0.6))
    next_velocity = float(np.clip(result[3] + acceleration * DT, 0.0, 0.8))
    mean_velocity = 0.5 * (result[3] + next_velocity)
    mid_yaw = result[2] + 0.5 * omega * DT
    result[0] += mean_velocity * math.cos(mid_yaw) * DT
    result[1] += mean_velocity * math.sin(mid_yaw) * DT
    result[2] = wrap_angle(result[2] + omega * DT)
    result[3] = next_velocity
    return result


def dynamic_pose(scenario_name, time_s):
    if scenario_name != "crossing_cart":
        return 1000.0, 1000.0
    return 3.8, max(-2.6, -0.5 - 0.3 * time_s)


def obb_aabb_collision(x, y, yaw, half_length, half_width, box_x, box_y, box_half):
    center_delta = np.asarray([box_x - x, box_y - y], dtype=np.float64)
    axes = (
        np.asarray([math.cos(yaw), math.sin(yaw)]),
        np.asarray([-math.sin(yaw), math.cos(yaw)]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
    )
    robot_axes = axes[:2]
    for axis in axes:
        robot_radius = (
            half_length * abs(float(np.dot(robot_axes[0], axis)))
            + half_width * abs(float(np.dot(robot_axes[1], axis)))
        )
        box_radius = box_half * (abs(axis[0]) + abs(axis[1]))
        if abs(float(np.dot(center_delta, axis))) > robot_radius + box_radius:
            return False
    return True


def supervise_swept_step(grid, current, proposed, scenario_name, time_s, samples=5):
    half_length = grid.half_length + grid.margin
    half_width = grid.half_width + grid.margin
    yaw_delta = wrap_angle(proposed[2] - current[2])
    for index in range(1, samples + 1):
        ratio = index / samples
        x = float(current[0] + ratio * (proposed[0] - current[0]))
        y = float(current[1] + ratio * (proposed[1] - current[1]))
        yaw = wrap_angle(float(current[2] + ratio * yaw_delta))
        sample_time = time_s + ratio * DT
        if grid.footprint_collision(x, y, yaw):
            return "abort_static_or_bounds"
        dynamic_x, dynamic_y = dynamic_pose(scenario_name, sample_time)
        if obb_aabb_collision(
            x,
            y,
            yaw,
            half_length,
            half_width,
            dynamic_x,
            dynamic_y,
            DYNAMIC_HALF_EXTENT_M,
        ):
            return "abort_dynamic_cart"
    return "allow"


def scenarios():
    return (
        Scenario("straight_aisle", (1.0, -2.0, 0.0, 0.0), (7.0, -2.0, 0.0)),
        Scenario("diagonal_turn", (1.0, -2.0, math.pi / 2.0, 0.0), (7.0, 2.0, 0.0)),
        Scenario("pallet_detour", (-4.0, 13.5, 0.0, 0.0), (6.5, 13.5, 0.0)),
        Scenario("crossing_cart", (1.0, -2.0, 0.0, 0.0), (7.0, -2.0, 0.0)),
    )


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile * 100.0))


def write_bev_evidence(path, decoded, occupied, valid, label):
    rgb = cv2.resize(decoded, (320, 240), interpolation=cv2.INTER_AREA)
    panel = np.full((*occupied.shape, 3), (32, 32, 32), dtype=np.uint8)
    panel[valid & (~occupied)] = (45, 170, 85)
    panel[valid & occupied] = (215, 65, 65)
    panel = cv2.resize(panel, (320, 240), interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack((rgb, panel))
    cv2.rectangle(canvas, (0, 0), (639, 28), (15, 15, 15), -1)
    cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"failed to write evidence: {path}")


def draw_trajectory(grid, scenario, planned, rows):
    canvas = np.full((*grid.raw.shape, 3), (245, 245, 245), dtype=np.uint8)
    canvas[grid.inflated > 0] = (190, 190, 190)
    canvas[grid.raw > 0] = (35, 35, 35)

    def pixel(point):
        row, col = grid.world_to_grid(point[0], point[1])
        return col, row

    planned_pixels = np.asarray([pixel(point) for point in planned], dtype=np.int32)
    actual_pixels = np.asarray([pixel((row["x_m"], row["y_m"])) for row in rows], dtype=np.int32)
    cv2.polylines(canvas, [planned_pixels], False, (220, 90, 20), 2, cv2.LINE_AA)
    if len(actual_pixels) >= 2:
        cv2.polylines(canvas, [actual_pixels], False, (20, 20, 220), 2, cv2.LINE_AA)
    if scenario.name == "crossing_cart":
        dynamic_points = [pixel(dynamic_pose(scenario.name, step * DT)) for step in range(250)]
        cv2.polylines(canvas, [np.asarray(dynamic_points, dtype=np.int32)], False, (20, 160, 160), 2, cv2.LINE_AA)
    cv2.circle(canvas, pixel(scenario.start), 4, (20, 160, 20), -1)
    cv2.circle(canvas, pixel(scenario.goal), 4, (180, 20, 180), -1)
    cv2.putText(canvas, scenario.name, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def run_scenario(
    scenario,
    grid,
    solver,
    model,
    sensor,
    bev,
    fixed_roi,
    render,
    output,
    max_steps_override=None,
):
    raw_path = astar(grid, scenario.start[:2], scenario.goal[:2])
    planned = prune_path(grid, raw_path)
    path = Polyline(planned)
    state = np.asarray(scenario.start, dtype=np.float64)
    previous_progress = 0.0
    rows = []
    abort_reason = None
    reached = False
    closest_dynamic = float("inf")
    dynamic_encounter_frames = 0
    max_steps = max_steps_override or max(800, int(math.ceil(path.length / 0.12 / DT)))
    telemetry_path = output / f"{scenario.name}.csv"
    with telemetry_path.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for step in range(max_steps):
            time_s = step * DT
            progress = max(previous_progress, path.nearest_s(state[0], state[1]))
            previous_progress = progress
            goal_distance = math.hypot(scenario.goal[0] - state[0], scenario.goal[1] - state[1])
            goal_yaw_error = wrap_angle(scenario.goal[2] - state[2])
            if goal_distance < 0.10 and abs(goal_yaw_error) < 0.08 and state[3] < 0.05:
                reached = True
                break
            dynamic_x, dynamic_y = dynamic_pose(scenario.name, time_s)
            render["dynamic_translate"].Set(render["Gf"].Vec3d(dynamic_x, dynamic_y, DYNAMIC_HALF_EXTENT_M))
            ext = sensor["body_extrinsics"]
            eye = (
                state[0] + math.cos(state[2]) * ext["forward_m"] - math.sin(state[2]) * ext["left_m"],
                state[1] + math.sin(state[2]) * ext["forward_m"] + math.cos(state[2]) * ext["left_m"],
                ext["height_m"],
            )
            render["camera_transform"].Set(
                camera_matrix(eye, state[2] + ext["yaw_rad"], ext["pitch_rad"], render["Gf"])
            )
            frame_started = time.perf_counter_ns()
            render_started = time.perf_counter_ns()
            render["rep"].orchestrator.step()
            render_ms = (time.perf_counter_ns() - render_started) / 1e6
            rgb = np.asarray(render["rgb"].get_data())[:, :, :3].astype(np.uint8)
            depth = np.asarray(render["depth"].get_data(), dtype=np.float32)
            pipeline_started = time.perf_counter_ns()
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                abort_reason = "abort_jpeg_encode"
                break
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            classes, _ = model.infer(decoded)
            semantic, observed = candidate_depth_lift(classes, depth, sensor, bev)
            occupied = occupancy_from_semantic(semantic)
            dummy = np.full(depth.shape, 13, dtype=np.uint8)
            _, depth_reference = candidate_depth_lift(dummy, depth, sensor, bev)
            reference_roi = fixed_roi & depth_reference
            valid_ratio = float(observed[reference_roi].mean()) if np.any(reference_roi) else 0.0
            obstacles, obstacle_count = bev_obstacle_parameters(occupied, observed, bev)
            mode, status, _, acceleration, omega, minimum_h = solve_shadow(
                solver, path, state, scenario.goal, obstacles
            )
            candidate_pipeline_ms = (time.perf_counter_ns() - pipeline_started) / 1e6
            if valid_ratio < 0.99:
                decision = "abort_perception_invalid"
            elif status != 0:
                decision = "abort_solver_failure"
            elif not all(math.isfinite(value) for value in (acceleration, omega)):
                decision = "abort_nonfinite_command"
            elif abs(acceleration) > 1.000001 or abs(omega) > 0.600001:
                decision = "abort_command_bounds"
            else:
                proposed = integrate_state(state, acceleration, omega)
                decision = supervise_swept_step(grid, state, proposed, scenario.name, time_s)
            command_applied = decision == "allow"
            proposed = integrate_state(state, acceleration, omega)
            static_collision = grid.footprint_collision(proposed[0], proposed[1], proposed[2])
            next_dynamic_x, next_dynamic_y = dynamic_pose(scenario.name, time_s + DT)
            dynamic_collision = obb_aabb_collision(
                proposed[0], proposed[1], proposed[2],
                grid.half_length + grid.margin,
                grid.half_width + grid.margin,
                next_dynamic_x, next_dynamic_y, DYNAMIC_HALF_EXTENT_M,
            )
            if command_applied:
                state = proposed
            dynamic_distance = math.hypot(state[0] - dynamic_x, state[1] - dynamic_y)
            if scenario.name == "crossing_cart" and dynamic_distance <= 2.2:
                dynamic_encounter_frames += 1
            if scenario.name == "crossing_cart" and dynamic_distance < closest_dynamic:
                closest_dynamic = dynamic_distance
                write_bev_evidence(
                    output / "crossing_cart_closest.png",
                    decoded,
                    occupied,
                    observed,
                    f"crossing_cart step={step} distance={dynamic_distance:.2f}m",
                )
            row = {
                "step": step, "time_s": time_s + DT,
                "x_m": float(state[0]), "y_m": float(state[1]),
                "yaw_rad": float(state[2]), "velocity_mps": float(state[3]),
                "acceleration_mps2": acceleration, "omega_radps": omega,
                "target_mode": mode, "goal_distance_m": goal_distance,
                "goal_yaw_error_rad": goal_yaw_error, "path_progress_m": progress,
                "path_error_m": path.distance_to(state[0], state[1]),
                "candidate_valid_ratio": valid_ratio,
                "candidate_obstacle_count": obstacle_count,
                "candidate_minimum_h": minimum_h, "solver_status": status,
                "render_ms": render_ms,
                "candidate_pipeline_ms": candidate_pipeline_ms,
                "sensor_to_command_ms": (time.perf_counter_ns() - frame_started) / 1e6,
                "dynamic_x_m": dynamic_x, "dynamic_y_m": dynamic_y,
                "dynamic_center_distance_m": dynamic_distance,
                "static_collision": int(static_collision),
                "dynamic_collision": int(dynamic_collision),
                "supervisor_decision": decision,
                "command_applied": int(command_applied),
                "candidate_controls_simulation": 1,
            }
            writer.writerow(row)
            target.flush()
            rows.append(row)
            if not command_applied:
                abort_reason = decision
                break
            if step % 100 == 99:
                print(
                    f"[Phase 5-G] {scenario.name} step={step + 1} "
                    f"goal={goal_distance:.2f}m v={state[3]:.2f} "
                    f"cmd=({acceleration:+.2f},{omega:+.2f})"
                )
    if not rows:
        raise RuntimeError(f"{scenario.name} produced no telemetry")
    terminal_position_error = math.hypot(scenario.goal[0] - state[0], scenario.goal[1] - state[1])
    terminal_yaw_error = abs(wrap_angle(scenario.goal[2] - state[2]))
    summary = {
        "name": scenario.name,
        "steps": len(rows),
        "duration_s": len(rows) * DT,
        "reached": reached,
        "abort_reason": abort_reason,
        "supervisor_aborts": int(abort_reason is not None),
        "solver_failures": sum(row["solver_status"] != 0 for row in rows),
        "static_collision_count": sum(row["static_collision"] for row in rows),
        "dynamic_collision_count": sum(row["dynamic_collision"] for row in rows),
        "command_applied_ratio": sum(row["command_applied"] for row in rows) / len(rows),
        "candidate_valid_ratio_mean": float(np.mean([row["candidate_valid_ratio"] for row in rows])),
        "candidate_pipeline_p95_ms": percentile([row["candidate_pipeline_ms"] for row in rows], 0.95),
        "sensor_to_command_p95_ms": percentile([row["sensor_to_command_ms"] for row in rows], 0.95),
        "render_p95_ms": percentile([row["render_ms"] for row in rows], 0.95),
        "path_error_p95_m": percentile([row["path_error_m"] for row in rows], 0.95),
        "terminal_position_error_m": terminal_position_error,
        "terminal_yaw_error_rad": terminal_yaw_error,
        "dynamic_encounter_frames": dynamic_encounter_frames,
        "minimum_dynamic_center_distance_m": closest_dynamic if scenario.name == "crossing_cart" else None,
        "telemetry": telemetry_path.name,
        "telemetry_sha256": sha256(telemetry_path),
    }
    return summary, draw_trajectory(grid, scenario, planned, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", choices=[item.name for item in scenarios()])
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    contract = json.loads((ROOT / "contracts/phase5/phase5g_contract.json").read_text())
    phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
    phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
    model_manifest = json.loads(MODEL_MANIFEST.read_text())
    model_path = ROOT / model_manifest["model"]
    if sha256(model_path) != model_manifest["model_sha256"]:
        raise SystemExit("warehouse candidate hash differs from manifest")
    manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    grid = OracleGrid(manifest_path.parent / manifest["archive"], manifest)
    selected = [item for item in scenarios() if args.scenario in (None, item.name)]
    output = args.output or ROOT / "artifacts/phase5g_takeover" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    sensor, bev = phase3["sensor_geometry"], phase3["bev_contract"]
    fixed_roi = control_roi(bev)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    summaries, evidence = [], []
    try:
        import omni
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils.semantics import add_labels
        from isaacsim.core.utils.stage import open_stage
        from pxr import Gf, UsdGeom

        open_stage(str(PHASE4_OVERLAY))
        stage = omni.usd.get_context().get_stage()
        stage.Load()
        obstacle = UsdGeom.Cube.Define(stage, "/Phase5GDynamicCart")
        obstacle.CreateSizeAttr(DYNAMIC_HALF_EXTENT_M * 2.0)
        obstacle.CreateDisplayColorAttr([(0.85, 0.08, 0.05)])
        dynamic_translate = UsdGeom.Xformable(obstacle).AddTranslateOp()
        add_labels(obstacle.GetPrim(), labels=["robot_or_cart"], taxonomy="class")
        camera = UsdGeom.Camera.Define(stage, "/Phase5GCamera")
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
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        rgb.attach([product])
        depth.attach([product])
        render = {
            "rep": rep, "Gf": Gf, "dynamic_translate": dynamic_translate,
            "camera_transform": camera_transform, "rgb": rgb, "depth": depth,
        }
        model = WarehouseCandidate(model_path)
        for _ in range(3):
            model.infer(np.zeros((480, 640, 3), dtype=np.uint8))
        for scenario in selected:
            solver = load_solver()
            summary, image = run_scenario(
                scenario, grid, solver, model, sensor, bev, fixed_roi, render,
                output, args.max_steps,
            )
            summaries.append(summary)
            evidence.append(image)
            print(
                f"{scenario.name}: reached={summary['reached']} "
                f"abort={summary['abort_reason']} steps={summary['steps']} "
                f"path_p95={summary['path_error_p95_m']:.3f}m"
            )
    except BaseException:
        app.close()
        raise
    evidence_path = output / "evidence.png"
    if not cv2.imwrite(str(evidence_path), np.concatenate(evidence, axis=1)):
        raise RuntimeError("failed to write Phase 5-G evidence")
    formal = len(selected) == len(contract["scenarios"]) and args.max_steps is None
    acceptance = contract["acceptance"]
    passed = bool(
        formal
        and len(summaries) == acceptance["required_scenarios"]
        and sum(item["reached"] for item in summaries) == acceptance["reached_scenarios"]
        and sum(item["supervisor_aborts"] for item in summaries) == acceptance["supervisor_aborts"]
        and sum(item["static_collision_count"] for item in summaries) == acceptance["static_collision_count"]
        and sum(item["dynamic_collision_count"] for item in summaries) == acceptance["dynamic_collision_count"]
        and sum(item["solver_failures"] for item in summaries) == acceptance["solver_failures"]
        and min(item["command_applied_ratio"] for item in summaries) >= acceptance["candidate_command_applied_ratio_min"]
        and min(item["candidate_valid_ratio_mean"] for item in summaries) >= acceptance["candidate_valid_ratio_mean_min"]
        and max(item["candidate_pipeline_p95_ms"] for item in summaries) <= acceptance["candidate_pipeline_p95_ms_max"]
        and max(item["path_error_p95_m"] for item in summaries) <= acceptance["path_error_p95_m_max"]
        and max(item["terminal_position_error_m"] for item in summaries) <= acceptance["terminal_position_error_m_max"]
        and max(item["terminal_yaw_error_rad"] for item in summaries) <= acceptance["terminal_yaw_error_rad_max"]
        and next(item for item in summaries if item["name"] == "crossing_cart")["dynamic_encounter_frames"]
        >= acceptance["dynamic_encounter_frames_min"]
    )
    result = {
        "schema_version": "phase5g-controlled-takeover-v1",
        "status": "takeover_gate_passed" if passed else ("takeover_gate_rejected" if formal else "smoke_only"),
        "candidate_controls_simulation": True,
        "oracle_role": "allow-or-abort swept collision supervisor",
        "oracle_command_override_count": 0,
        "real_vehicle_control_allowed": False,
        "scenarios": summaries,
        "acceptance": acceptance,
        "evidence": evidence_path.name,
        "evidence_sha256": sha256(evidence_path),
        "gate_passed": passed,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2))
    print(f"Phase 5-G artifacts: {output}")
    if formal and not passed:
        raise SystemExit(1)
    app.close()


if __name__ == "__main__":
    main()
