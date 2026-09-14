#!/usr/bin/env python3


SCENARIOS = ["straight_aisle", "diagonal_turn", "pallet_detour", "crossing_cart"]


def hour_gate(metrics, required_frames, required_duration_s, active_ratio_min, latency_p95_ms_max):
    return bool(
        int(metrics["frames"]) == int(required_frames)
        and float(metrics["wall_duration_s"]) >= float(required_duration_s)
        and metrics["exact_source_frame_ids"]
        and metrics["scenario_coverage"] == SCENARIOS
        and int(metrics["collision_count"]) == 0
        and int(metrics["wrong_generation_commands"]) == 0
        and float(metrics["active_ratio"]) >= float(active_ratio_min)
        and float(metrics["sensor_to_wheel_p95_ms"]) <= float(latency_p95_ms_max)
    )


def out_of_band_stop_gate(receipt, ledger, watchdog_ms):
    latency_ms = (int(ledger["monotonic_ns"]) - int(receipt["kill_monotonic_ns"])) / 1e6
    return bool(
        receipt["confirmed"]
        and float(ledger["linear"]) == 0.0
        and float(ledger["angular"]) == 0.0
        and ledger["reason"] in {"coordinator_fail_safe_zero", "watchdog_zero", "terminal_zero"}
        and 0.0 <= latency_ms <= float(watchdog_ms)
    )


def recovery_gate(metrics, stop_latency_frames_max):
    return bool(
        metrics["fault_observed"]
        and int(metrics["maximum_fault_stop_latency_frames"]) <= int(stop_latency_frames_max)
        and int(metrics["reset_events"]) >= 1
        and metrics["active_after_last_reset"]
        and int(metrics["collision_count"]) == 0
    )


def resource_receipt_gate(run_mode, receipt):
    if run_mode == "gpu_oom_recovery":
        free_before = int(receipt.get("free_bytes_before", 0))
        free_after = int(receipt.get("free_bytes_after_release", 0))
        return bool(
            receipt.get("observed")
            and receipt.get("message_contains_cuda_oom")
            and int(receipt.get("allocated_bytes_before_failure", 0)) > 0
            and free_before > 0
            and free_after >= 0.9 * free_before
        )
    if run_mode == "disk_full_recovery":
        return not receipt.get("written", True) and int(receipt.get("errno", -1)) == 28
    raise ValueError(f"unsupported resource fault mode: {run_mode}")


def coordinator_gate(metrics, receipt, ledger, watchdog_ms):
    return bool(
        receipt.get("target") == "coordinator"
        and metrics["exact_source_frame_ids"]
        and int(metrics["frames"]) > int(receipt.get("source_frame_id", -1))
        and int(metrics["collision_count"]) == 0
        and int(metrics["wrong_generation_commands"]) == 0
        and ledger.get("reason") == "coordinator_fail_safe_zero"
        and out_of_band_stop_gate(receipt, ledger, watchdog_ms)
    )


def daemon_gate(metrics, receipt, ledger, watchdog_ms):
    return bool(
        receipt.get("target") == "daemon"
        and metrics["exact_source_frame_ids"]
        and int(metrics["frames"]) > 0
        and int(metrics["collision_count"]) == 0
        and int(metrics["wrong_generation_commands"]) == 0
        and out_of_band_stop_gate(receipt, ledger, watchdog_ms)
    )
