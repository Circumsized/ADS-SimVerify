#!/usr/bin/env python3
import copy
import math
import random


def build_matrix(contract):
    matrix = contract["matrix"]
    seeds = matrix["seeds"]
    profiles = matrix["profiles"]
    if len(seeds) != len(profiles):
        raise ValueError("Phase 6 requires exactly one perturbation profile per seed")
    position_limit = float(matrix["start_goal_limits"]["position_m"])
    yaw_limit = math.radians(float(matrix["start_goal_limits"]["yaw_deg"]))
    cells = []
    for seed_index, (seed, profile) in enumerate(zip(seeds, profiles), start=1):
        rng = random.Random(int(seed))
        for scenario in matrix["scenarios"]:
            start_goal = {
                "start_dx_m": rng.uniform(-position_limit, position_limit),
                "start_dy_m": rng.uniform(-position_limit, position_limit),
                "start_dyaw_rad": rng.uniform(-yaw_limit, yaw_limit),
                "goal_dx_m": rng.uniform(-position_limit, position_limit),
                "goal_dy_m": rng.uniform(-position_limit, position_limit),
                "goal_dyaw_rad": rng.uniform(-yaw_limit, yaw_limit),
            }
            cells.append({
                "case_id": f"s{seed_index:02d}_{scenario}",
                "seed": int(seed),
                "scenario": scenario,
                "profile": profile["name"],
                "perturbations": {
                    "start_goal": start_goal,
                    "lighting": {"intensity_scale": profile["lighting_intensity_scale"]},
                    "material": {"dynamic_obstacle_rgb": copy.deepcopy(profile["dynamic_obstacle_rgb"])},
                    "jpeg_rgb": copy.deepcopy(profile["jpeg_rgb"]),
                    "metric_depth": copy.deepcopy(profile["metric_depth"]),
                    "camera_extrinsics": copy.deepcopy(profile["camera_delta"]),
                    "dynamic_obstacle": copy.deepcopy(profile["dynamic_obstacle"]),
                },
            })
    return cells


def aggregate_matrix(results, inherited, acceptance):
    case_ids = [item["case_id"] for item in results]
    reached_cases = sum(bool(item["reached"]) for item in results)
    collisions = sum(
        int(item["static_collision_count"]) + int(item["dynamic_collision_count"])
        for item in results
    )
    aggregate = {
        "cases": len(results),
        "unique_cases": len(set(case_ids)),
        "reached_cases": reached_cases,
        "goal_reach_rate": reached_cases / len(results) if results else 0.0,
        "collision_count": collisions,
        "supervisor_aborts": sum(int(item["supervisor_aborts"]) for item in results),
        "solver_failures": sum(int(item["solver_failures"]) for item in results),
        "candidate_valid_ratio_mean_min": min(
            (float(item["candidate_valid_ratio_mean"]) for item in results), default=0.0
        ),
        "sensor_to_wheel_p95_ms_max": max(
            (float(item["sensor_to_wheel_p95_ms"]) for item in results), default=float("inf")
        ),
        "path_error_p95_m_max": max(
            (float(item["path_error_p95_m"]) for item in results), default=float("inf")
        ),
        "terminal_position_error_m_max": max(
            (float(item["terminal_position_error_m"]) for item in results), default=float("inf")
        ),
        "terminal_yaw_error_rad_max": max(
            (float(item["terminal_yaw_error_rad"]) for item in results), default=float("inf")
        ),
        "wheel_command_delta_p95_max": max(
            (float(item["wheel_command_delta_p95"]) for item in results), default=float("inf")
        ),
        "inherited": dict(inherited),
    }
    aggregate["gate_passed"] = bool(
        aggregate["cases"] == acceptance["required_cases"]
        and aggregate["unique_cases"] == acceptance["required_cases"]
        and aggregate["goal_reach_rate"] >= acceptance["goal_reach_rate_min"]
        and aggregate["collision_count"] <= acceptance["collision_count_max"]
        and aggregate["supervisor_aborts"] <= acceptance["supervisor_aborts_max"]
        and aggregate["solver_failures"] <= acceptance["solver_failures_max"]
        and aggregate["candidate_valid_ratio_mean_min"] >= acceptance["candidate_valid_ratio_mean_min"]
        and aggregate["sensor_to_wheel_p95_ms_max"] <= acceptance["sensor_to_wheel_p95_ms_max"]
        and aggregate["path_error_p95_m_max"] <= acceptance["path_error_p95_m_max"]
        and aggregate["terminal_position_error_m_max"] <= acceptance["terminal_position_error_m_max"]
        and aggregate["terminal_yaw_error_rad_max"] <= acceptance["terminal_yaw_error_rad_max"]
        and aggregate["wheel_command_delta_p95_max"] <= acceptance["wheel_command_delta_p95_max"]
        and inherited["hour_frames"] >= acceptance["inherited_hour_frames_min"]
        and inherited["hour_duration_s"] >= acceptance["inherited_hour_duration_s_min"]
        and inherited["hour_collision_count"] <= acceptance["inherited_hour_collision_count_max"]
        and inherited["maximum_fault_recovery_frames"] <= acceptance["inherited_fault_recovery_frames_max"]
    )
    return aggregate
