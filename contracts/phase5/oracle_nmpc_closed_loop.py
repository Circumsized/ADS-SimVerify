#!/usr/bin/env python3
import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM_ENV = ROOT / "simulation-env"
ACADOS_SOURCE = SIM_ENV / "acados"
N = 20
DT = 0.05
MAX_SPEED_MPS = 0.50


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Scenario:
    name: str
    start: tuple
    goal: tuple


class OracleGrid:
    def __init__(self, archive, manifest):
        arrays = np.load(archive)
        self.raw = arrays["raw_occupied"]
        self.inflated = arrays["inflated_occupied"]
        self.clearance = arrays["clearance_m"]
        self.min_x, self.max_x, self.min_y, self.max_y = manifest["bounds_xy_m"]
        self.resolution = float(manifest["resolution_m"])
        footprint = manifest["robot_footprint"]
        self.half_length = float(footprint["half_length_m"])
        self.half_width = float(footprint["half_width_m"])
        self.margin = float(footprint["safety_margin_m"])
        rows, cols = np.nonzero(self.raw)
        self.obstacle_x = self.min_x + cols.astype(np.float64) * self.resolution
        self.obstacle_y = self.max_y - rows.astype(np.float64) * self.resolution

    def world_to_grid(self, x, y):
        row = int(round((self.max_y - y) / self.resolution))
        col = int(round((x - self.min_x) / self.resolution))
        return row, col

    def grid_to_world(self, row, col):
        return (
            self.min_x + col * self.resolution,
            self.max_y - row * self.resolution,
        )

    def inside(self, row, col):
        return 0 <= row < self.raw.shape[0] and 0 <= col < self.raw.shape[1]

    def center_clearance(self, x, y):
        row, col = self.world_to_grid(x, y)
        if not self.inside(row, col):
            return 0.0
        return float(self.clearance[row, col])

    def footprint_collision(self, x, y, yaw):
        half_l = self.half_length + self.margin
        half_w = self.half_width + self.margin
        spacing = self.resolution * 0.5
        local_x = np.arange(-half_l, half_l + spacing * 0.5, spacing)
        local_y = np.arange(-half_w, half_w + spacing * 0.5, spacing)
        xx, yy = np.meshgrid(local_x, local_y)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        world_x = x + xx.ravel() * cos_yaw - yy.ravel() * sin_yaw
        world_y = y + xx.ravel() * sin_yaw + yy.ravel() * cos_yaw
        rows = np.rint((self.max_y - world_y) / self.resolution).astype(np.int64)
        cols = np.rint((world_x - self.min_x) / self.resolution).astype(np.int64)
        inside = (
            (rows >= 0)
            & (rows < self.raw.shape[0])
            & (cols >= 0)
            & (cols < self.raw.shape[1])
        )
        if not bool(np.all(inside)):
            return True
        return bool(np.any(self.raw[rows, cols]))

    def local_obstacles(self, x, y, yaw):
        dx = self.obstacle_x - x
        dy = self.obstacle_y - y
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        angle = np.arctan2(local_y, local_x)
        distance2 = local_x * local_x + local_y * local_y
        sectors = (
            (-2.10, -0.35),
            (-0.35, 0.35),
            (0.35, 2.10),
        )
        obstacles = []
        for low, high in sectors:
            mask = (angle >= low) & (angle < high) & (local_x > -0.10) & (distance2 < 2.5**2)
            indices = np.flatnonzero(mask)
            if indices.size:
                index = indices[np.argmin(distance2[indices])]
                obstacles.extend((float(local_x[index]), float(local_y[index]), 0.30, 0.30))
            else:
                obstacles.extend((1000.0, 1000.0, 0.10, 0.10))
        return np.asarray(obstacles, dtype=np.float64)


def astar(grid, start_xy, goal_xy):
    start = grid.world_to_grid(*start_xy)
    goal = grid.world_to_grid(*goal_xy)
    if not grid.inside(*start) or not grid.inside(*goal):
        raise ValueError("start or goal lies outside the oracle map")
    if grid.inflated[start] or grid.inflated[goal]:
        raise ValueError("start or goal lies inside inflated oracle occupancy")
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    queue = [(0.0, 0.0, start)]
    cost = {start: 0.0}
    parent = {}
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost > cost[current] + 1e-9:
            continue
        if current == goal:
            cells = [current]
            while cells[-1] != start:
                cells.append(parent[cells[-1]])
            cells.reverse()
            return [grid.grid_to_world(*cell) for cell in cells]
        row, col = current
        for d_row, d_col, step_cost in moves:
            neighbor = row + d_row, col + d_col
            if not grid.inside(*neighbor) or grid.inflated[neighbor]:
                continue
            if d_row and d_col and (grid.inflated[row, col + d_col] or grid.inflated[row + d_row, col]):
                continue
            candidate = current_cost + step_cost
            if candidate + 1e-9 >= cost.get(neighbor, float("inf")):
                continue
            cost[neighbor] = candidate
            parent[neighbor] = current
            heuristic = math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
            heapq.heappush(queue, (candidate + heuristic, candidate, neighbor))
    raise RuntimeError("A* found no path in inflated oracle occupancy")


def line_of_sight(grid, start, end):
    distance = math.dist(start, end)
    samples = max(2, int(math.ceil(distance / (grid.resolution * 0.5))) + 1)
    for t in np.linspace(0.0, 1.0, samples):
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        row, col = grid.world_to_grid(x, y)
        if not grid.inside(row, col) or grid.inflated[row, col]:
            return False
    return True


def prune_path(grid, points):
    if len(points) <= 2:
        return points
    pruned = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        candidate = len(points) - 1
        while candidate > anchor + 1 and not line_of_sight(grid, points[anchor], points[candidate]):
            candidate -= 1
        pruned.append(points[candidate])
        anchor = candidate
    return pruned


class Polyline:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)
        lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("path contains duplicate points")
        self.segment_lengths = lengths
        self.cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        self.length = float(self.cumulative[-1])

    def nearest_s(self, x, y):
        point = np.array([x, y], dtype=np.float64)
        starts = self.points[:-1]
        vectors = self.points[1:] - starts
        ratios = np.sum((point - starts) * vectors, axis=1) / (self.segment_lengths**2)
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = starts + ratios[:, None] * vectors
        index = int(np.argmin(np.sum((projections - point) ** 2, axis=1)))
        return float(self.cumulative[index] + ratios[index] * self.segment_lengths[index])

    def distance_to(self, x, y):
        point = np.array([x, y], dtype=np.float64)
        starts = self.points[:-1]
        vectors = self.points[1:] - starts
        ratios = np.sum((point - starts) * vectors, axis=1) / (self.segment_lengths**2)
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = starts + ratios[:, None] * vectors
        return float(np.sqrt(np.min(np.sum((projections - point) ** 2, axis=1))))

    def pose(self, distance, goal_yaw):
        distance = float(np.clip(distance, 0.0, self.length))
        index = min(int(np.searchsorted(self.cumulative, distance, side="right") - 1), len(self.segment_lengths) - 1)
        ratio = (distance - self.cumulative[index]) / self.segment_lengths[index]
        point = self.points[index] + ratio * (self.points[index + 1] - self.points[index])
        vector = self.points[index + 1] - self.points[index]
        heading = math.atan2(vector[1], vector[0])
        remaining = self.length - distance
        if remaining < 0.60:
            blend = 1.0 - remaining / 0.60
            heading = wrap_angle(heading + blend * wrap_angle(goal_yaw - heading))
        return float(point[0]), float(point[1]), heading


def load_solver():
    os.environ.setdefault("ACADOS_SOURCE_DIR", str(ACADOS_SOURCE))
    sys.path.insert(0, str(ACADOS_SOURCE / "interfaces/acados_template"))
    from acados_template import AcadosOcpSolver

    return AcadosOcpSolver(
        None,
        json_file=str(SIM_ENV / "acados_ocp.json"),
        generate=False,
        build=False,
    )


def target_mode(path, state, progress, goal):
    x, y, yaw, _ = state
    goal_x, goal_y, goal_yaw = goal
    goal_distance = math.hypot(goal_x - x, goal_y - y)
    _, _, path_heading = path.pose(min(progress + 0.20, path.length), goal_yaw)
    heading_error = wrap_angle(path_heading - yaw)
    goal_yaw_error = wrap_angle(goal_yaw - yaw)
    if goal_distance < 0.10 and abs(goal_yaw_error) > 0.08:
        return "goal_yaw", 0.0, goal_yaw_error
    if abs(heading_error) > math.radians(35.0):
        return "path_yaw", 0.0, heading_error
    _, _, near_heading = path.pose(min(progress + 0.20, path.length), goal_yaw)
    _, _, far_heading = path.pose(min(progress + 0.80, path.length), goal_yaw)
    turn = abs(wrap_angle(far_heading - near_heading))
    speed = 0.25 if turn > math.radians(18.0) else MAX_SPEED_MPS
    speed = min(speed, max(0.10, 0.65 * goal_distance))
    return "track", speed, heading_error


def set_nmpc_problem(solver, path, state, progress, goal, speed, mode, heading_error, obstacle_parameters):
    x, y, yaw, velocity = state
    initial = np.array([0.0, 0.0, 0.0, velocity], dtype=np.float64)
    solver.constraints_set(0, "lbx", initial)
    solver.constraints_set(0, "ubx", initial)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    for stage in range(N + 1):
        if mode == "track":
            ref_s = min(path.length, progress + speed * DT * stage)
            ref_x, ref_y, ref_yaw = path.pose(ref_s, goal[2])
            dx, dy = ref_x - x, ref_y - y
            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
            local_yaw = wrap_angle(ref_yaw - yaw)
            ref_speed = 0.0 if ref_s >= path.length - 1e-6 else speed
        else:
            local_x, local_y, local_yaw, ref_speed = 0.0, 0.0, heading_error, 0.0
        if stage < N:
            solver.set(stage, "yref", np.array([local_x, local_y, local_yaw, ref_speed, 0.0, 0.0]))
        else:
            solver.set(stage, "yref", np.array([local_x, local_y, local_yaw, ref_speed]))
        solver.set(stage, "p", obstacle_parameters)


def minimum_predicted_obstacle_h(solver, obstacle_parameters):
    minimum = float("inf")
    for stage in range(N + 1):
        predicted = solver.get(stage, "x")
        for index in range(3):
            obs_x, obs_y, axis_a, axis_b = obstacle_parameters[index * 4 : index * 4 + 4]
            h_value = (
                (float(predicted[0]) - obs_x) ** 2 / (axis_a**2 + 1e-6)
                + (float(predicted[1]) - obs_y) ** 2 / (axis_b**2 + 1e-6)
            )
            minimum = min(minimum, h_value)
    return minimum


def draw_evidence(grid, scenario, planned, telemetry):
    canvas = np.full((*grid.raw.shape, 3), (245, 245, 245), dtype=np.uint8)
    canvas[grid.inflated > 0] = (190, 190, 190)
    canvas[grid.raw > 0] = (35, 35, 35)

    def pixel(point):
        row, col = grid.world_to_grid(point[0], point[1])
        return col, row

    planned_pixels = np.asarray([pixel(point) for point in planned], dtype=np.int32)
    actual_pixels = np.asarray([pixel((row["x_m"], row["y_m"])) for row in telemetry], dtype=np.int32)
    cv2.polylines(canvas, [planned_pixels], False, (220, 90, 20), 2, cv2.LINE_AA)
    if len(actual_pixels) >= 2:
        cv2.polylines(canvas, [actual_pixels], False, (20, 20, 220), 2, cv2.LINE_AA)
    cv2.circle(canvas, pixel(scenario.start), 4, (20, 160, 20), -1)
    cv2.circle(canvas, pixel(scenario.goal), 4, (180, 20, 180), -1)
    cv2.putText(canvas, scenario.name, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def run_scenario(grid, solver, scenario, output):
    raw_path = astar(grid, scenario.start[:2], scenario.goal[:2])
    planned = prune_path(grid, raw_path)
    path = Polyline(planned)
    state = np.array(scenario.start, dtype=np.float64)
    telemetry = []
    solver_failures = 0
    collision_count = 0
    reached = False
    max_steps = max(500, int(math.ceil(path.length / 0.18 / DT)))
    previous_progress = 0.0

    for step in range(max_steps):
        progress = max(previous_progress, path.nearest_s(state[0], state[1]))
        previous_progress = progress
        goal_distance = math.hypot(scenario.goal[0] - state[0], scenario.goal[1] - state[1])
        goal_yaw_error = wrap_angle(scenario.goal[2] - state[2])
        if goal_distance < 0.10 and abs(goal_yaw_error) < 0.08 and state[3] < 0.05:
            reached = True
            break
        mode, speed, heading_error = target_mode(path, state, progress, scenario.goal)
        obstacle_parameters = grid.local_obstacles(state[0], state[1], state[2])
        set_nmpc_problem(
            solver,
            path,
            state,
            progress,
            scenario.goal,
            speed,
            mode,
            heading_error,
            obstacle_parameters,
        )
        started = time.perf_counter_ns()
        status = int(solver.solve())
        solve_ms = (time.perf_counter_ns() - started) / 1e6
        if status == 0:
            acceleration, omega = (float(value) for value in solver.get(0, "u"))
            minimum_obstacle_h = minimum_predicted_obstacle_h(solver, obstacle_parameters)
        else:
            solver_failures += 1
            acceleration, omega = -1.0, 0.0
            minimum_obstacle_h = 0.0
        acceleration = float(np.clip(acceleration, -1.0, 1.0))
        omega = float(np.clip(omega, -0.6, 0.6))
        next_velocity = float(np.clip(state[3] + acceleration * DT, 0.0, 0.8))
        mean_velocity = 0.5 * (state[3] + next_velocity)
        mid_yaw = state[2] + 0.5 * omega * DT
        state[0] += mean_velocity * math.cos(mid_yaw) * DT
        state[1] += mean_velocity * math.sin(mid_yaw) * DT
        state[2] = wrap_angle(state[2] + omega * DT)
        state[3] = next_velocity
        collision = grid.footprint_collision(state[0], state[1], state[2])
        collision_count += int(collision)
        telemetry.append(
            {
                "step": step,
                "time_s": (step + 1) * DT,
                "x_m": float(state[0]),
                "y_m": float(state[1]),
                "yaw_rad": float(state[2]),
                "velocity_mps": float(state[3]),
                "acceleration_mps2": acceleration,
                "omega_radps": omega,
                "goal_distance_m": goal_distance,
                "goal_yaw_error_rad": goal_yaw_error,
                "path_progress_m": progress,
                "path_error_m": path.distance_to(state[0], state[1]),
                "center_clearance_m": grid.center_clearance(state[0], state[1]),
                "minimum_predicted_obstacle_h": minimum_obstacle_h,
                "solve_ms": solve_ms,
                "solver_status": status,
                "mode": mode,
                "collision": int(collision),
            }
        )
        if collision or solver_failures > 5:
            break

    terminal_position_error = math.hypot(scenario.goal[0] - state[0], scenario.goal[1] - state[1])
    terminal_yaw_error = abs(wrap_angle(scenario.goal[2] - state[2]))
    solve_times = np.asarray([row["solve_ms"] for row in telemetry], dtype=np.float64)
    clearances = np.asarray([row["center_clearance_m"] for row in telemetry], dtype=np.float64)
    path_errors = np.asarray([row["path_error_m"] for row in telemetry], dtype=np.float64)
    obstacle_h = np.asarray([row["minimum_predicted_obstacle_h"] for row in telemetry], dtype=np.float64)
    summary = {
        "name": scenario.name,
        "start": list(scenario.start),
        "goal": list(scenario.goal),
        "raw_astar_points": len(raw_path),
        "pruned_path_points": len(planned),
        "planned_length_m": path.length,
        "steps": len(telemetry),
        "duration_s": len(telemetry) * DT,
        "reached": reached,
        "collision_count": collision_count,
        "solver_failures": solver_failures,
        "solve_latency_ms": {
            "mean": float(solve_times.mean()) if solve_times.size else None,
            "p95": float(np.percentile(solve_times, 95)) if solve_times.size else None,
            "max": float(solve_times.max()) if solve_times.size else None,
        },
        "minimum_center_clearance_m": float(clearances.min()) if clearances.size else None,
        "path_error_m": {
            "mean": float(path_errors.mean()) if path_errors.size else None,
            "p95": float(np.percentile(path_errors, 95)) if path_errors.size else None,
            "max": float(path_errors.max()) if path_errors.size else None,
        },
        "minimum_predicted_obstacle_h": float(obstacle_h.min()) if obstacle_h.size else None,
        "terminal_position_error_m": terminal_position_error,
        "terminal_yaw_error_rad": terminal_yaw_error,
        "max_abs_omega_radps": max((abs(row["omega_radps"]) for row in telemetry), default=0.0),
        "max_abs_acceleration_mps2": max((abs(row["acceleration_mps2"]) for row in telemetry), default=0.0),
    }
    summary["passed"] = bool(
        reached
        and collision_count == 0
        and solver_failures == 0
        and summary["solve_latency_ms"]["p95"] <= 50.0
        and summary["path_error_m"]["p95"] <= 0.20
        and summary["minimum_predicted_obstacle_h"] >= 0.98
        and summary["max_abs_omega_radps"] <= 0.600001
        and summary["max_abs_acceleration_mps2"] <= 1.000001
        and terminal_position_error <= 0.10
        and terminal_yaw_error <= 0.08
    )
    csv_path = output / f"{scenario.name}.csv"
    with csv_path.open("w", newline="", encoding="ascii") as target:
        writer = csv.DictWriter(target, fieldnames=list(telemetry[0]))
        writer.writeheader()
        writer.writerows(telemetry)
    summary["telemetry"] = csv_path.name
    summary["telemetry_sha256"] = sha256(csv_path)
    return summary, draw_evidence(grid, scenario, planned, telemetry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.oracle:
        oracle = args.oracle.resolve()
    else:
        candidates = sorted((ROOT / "artifacts/phase5a_oracle").glob("*/manifest.json"))
        if not candidates:
            raise SystemExit("No Phase 5-A oracle map found; run build_oracle_map.py first")
        oracle = candidates[-1].parent
    output = args.output or ROOT / "artifacts/phase5a_closed_loop" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = oracle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = oracle / manifest["archive"]
    if sha256(archive) != manifest["archive_sha256"]:
        raise SystemExit("Oracle map archive hash mismatch")
    grid = OracleGrid(archive, manifest)
    scenarios = (
        Scenario("straight_aisle", (1.0, -2.0, 0.0, 0.0), (7.0, -2.0, 0.0)),
        Scenario("diagonal_turn", (1.0, -2.0, math.pi / 2.0, 0.0), (7.0, 2.0, 0.0)),
        Scenario("pallet_detour", (-4.0, 13.5, 0.0, 0.0), (6.5, 13.5, 0.0)),
    )
    solver = load_solver()
    summaries = []
    evidence = []
    for scenario in scenarios:
        scenario_summary, scenario_evidence = run_scenario(grid, solver, scenario, output)
        summaries.append(scenario_summary)
        evidence.append(scenario_evidence)
        print(
            f"{scenario.name}: passed={scenario_summary['passed']} "
            f"reached={scenario_summary['reached']} collision={scenario_summary['collision_count']} "
            f"solver_failures={scenario_summary['solver_failures']} "
            f"position_error={scenario_summary['terminal_position_error_m']:.3f}m "
            f"p95={scenario_summary['solve_latency_ms']['p95']:.3f}ms"
        )
    combined = np.concatenate(evidence, axis=1)
    evidence_path = output / "evidence.png"
    if not cv2.imwrite(str(evidence_path), combined):
        raise RuntimeError("failed to write Phase 5-A evidence image")
    result = {
        "schema_version": "phase5a-oracle-nmpc-v1",
        "status": "passed" if all(item["passed"] for item in summaries) else "failed",
        "control_source": "USD oracle occupancy only",
        "learned_perception_in_control_loop": False,
        "solver": "simulation-env/acados_ocp.json + generated acados C solver",
        "control_rate_hz": int(round(1.0 / DT)),
        "oracle_manifest": str(manifest_path.relative_to(ROOT)),
        "oracle_manifest_sha256": sha256(manifest_path),
        "oracle_archive_sha256": sha256(archive),
        "acceptance": {
            "required_scenarios": 3,
            "collision_count": 0,
            "solver_failures": 0,
            "solve_p95_ms_max": 50.0,
            "path_error_p95_m_max": 0.20,
            "minimum_predicted_obstacle_h": 0.98,
            "max_abs_omega_radps": 0.600001,
            "max_abs_acceleration_mps2": 1.000001,
            "terminal_position_error_m_max": 0.10,
            "terminal_yaw_error_rad_max": 0.08,
        },
        "scenarios": summaries,
        "evidence": evidence_path.name,
        "evidence_sha256": sha256(evidence_path),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(f"Phase 5-A result: {result['status']}")
    print(f"Evidence: {output}")
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
