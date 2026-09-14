#!/usr/bin/env python3
import math


DEFAULT_THRESHOLDS = {
    "camera": {
        "paired_frame_ratio_min": 0.99,
        "depth_scale_relative_error_max": 0.02,
    },
    "localization": {
        "duration_s_min": 600.0,
        "ate_rmse_m_max": 0.10,
        "rpe_translation_p95_m_max": 0.08,
        "yaw_error_p95_deg_max": 3.0,
        "dropout_ratio_max": 0.01,
        "relocalization_success_ratio_min": 0.95,
        "relocalization_p95_s_max": 2.0,
    },
    "global_planning": {
        "routes_min": 8,
        "route_success_ratio_min": 1.0,
        "invalid_or_occupied_waypoints_max": 0,
    },
    "collision_supervisor": {
        "physical_cases_min": 40,
        "stop_recall_min": 0.99,
        "go_specificity_min": 0.95,
        "minimum_stop_margin_m_min": 0.10,
    },
    "actuator": {
        "velocity_tracking_p95_mps_max": 0.08,
        "yaw_rate_tracking_p95_radps_max": 0.12,
        "watchdog_stop_p95_ms_max": 150.0,
        "emergency_stop_p95_ms_max": 150.0,
        "zero_command_creep_mps_max": 0.01,
    },
}


def audit_topology_text(text):
    lowered = text.lower()
    tokens = ("isaac_sim_env", "simulation-env", "usd", "oracle")
    found = [token for token in tokens if token in lowered]
    return {
        "forbidden_runtime_dependencies": found,
        "gate_passed": not found,
    }


def finite_number(block, key):
    value = block.get(key)
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def at_least(block, key, limit):
    return finite_number(block, key) and float(block[key]) >= float(limit)


def at_most(block, key, limit):
    return finite_number(block, key) and float(block[key]) <= float(limit)


def evaluate_acceptance(evidence, thresholds=DEFAULT_THRESHOLDS):
    results = {}

    block = evidence.get("topology", {})
    results["topology"] = bool(
        block
        and block.get("forbidden_runtime_dependencies") == []
        and block.get("global_map_source") == "real_slam_occupancy_map"
        and block.get("independent_safety_source") == "independent_metric_depth_guard"
    )

    block = evidence.get("camera", {})
    gate = thresholds["camera"]
    results["camera"] = bool(
        block.get("device_present")
        and block.get("calibration_gate_passed")
        and at_least(block, "paired_frame_ratio", gate["paired_frame_ratio_min"])
        and at_most(
            block, "depth_scale_relative_error", gate["depth_scale_relative_error_max"]
        )
    )

    block = evidence.get("localization", {})
    gate = thresholds["localization"]
    results["localization"] = bool(
        at_least(block, "duration_s", gate["duration_s_min"])
        and at_most(block, "ate_rmse_m", gate["ate_rmse_m_max"])
        and at_most(block, "rpe_translation_p95_m", gate["rpe_translation_p95_m_max"])
        and at_most(block, "yaw_error_p95_deg", gate["yaw_error_p95_deg_max"])
        and at_most(block, "dropout_ratio", gate["dropout_ratio_max"])
        and at_least(
            block,
            "relocalization_success_ratio",
            gate["relocalization_success_ratio_min"],
        )
        and at_most(block, "relocalization_p95_s", gate["relocalization_p95_s_max"])
    )

    block = evidence.get("global_planning", {})
    gate = thresholds["global_planning"]
    results["global_planning"] = bool(
        block.get("map_hash_present")
        and at_least(block, "routes", gate["routes_min"])
        and at_least(block, "route_success_ratio", gate["route_success_ratio_min"])
        and at_most(
            block,
            "invalid_or_occupied_waypoints",
            gate["invalid_or_occupied_waypoints_max"],
        )
    )

    block = evidence.get("collision_supervisor", {})
    gate = thresholds["collision_supervisor"]
    results["collision_supervisor"] = bool(
        block.get("oracle_used") is False
        and at_least(block, "physical_cases", gate["physical_cases_min"])
        and at_least(block, "stop_recall", gate["stop_recall_min"])
        and at_least(block, "go_specificity", gate["go_specificity_min"])
        and at_least(
            block, "minimum_stop_margin_m", gate["minimum_stop_margin_m_min"]
        )
    )

    block = evidence.get("actuator", {})
    gate = thresholds["actuator"]
    results["actuator"] = bool(
        block.get("wheels_off_ground_tested")
        and block.get("command_sign_correct")
        and at_most(
            block, "velocity_tracking_p95_mps", gate["velocity_tracking_p95_mps_max"]
        )
        and at_most(
            block,
            "yaw_rate_tracking_p95_radps",
            gate["yaw_rate_tracking_p95_radps_max"],
        )
        and at_most(block, "watchdog_stop_p95_ms", gate["watchdog_stop_p95_ms_max"])
        and at_most(
            block, "emergency_stop_p95_ms", gate["emergency_stop_p95_ms_max"]
        )
        and at_most(block, "zero_command_creep_mps", gate["zero_command_creep_mps_max"])
    )

    blocked = [name for name, passed in results.items() if not passed]
    passed = not blocked
    return {
        "gates": results,
        "blocked_gates": blocked,
        "gate_passed": passed,
        "real_vehicle_control_allowed": passed,
    }
