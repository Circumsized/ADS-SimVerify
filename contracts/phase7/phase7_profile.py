#!/usr/bin/env python3
import math

import numpy as np


def timing_summary(values):
    samples = np.asarray(values, dtype=np.float64)
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("timing samples must be non-empty and finite")
    return {
        "samples": int(samples.size),
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50.0)),
        "p95_ms": float(np.percentile(samples, 95.0)),
        "max_ms": float(np.max(samples)),
    }


def stream_bandwidth(streams):
    result = {}
    total = 0.0
    for name, stream in streams.items():
        elements = math.prod(int(value) for value in stream["shape"])
        mbps = elements * int(stream["bytes_per_element"]) * float(stream["rate_hz"]) / 1e6
        result[f"{name}_mbps"] = mbps
        total += mbps
    result["total_mbps"] = total
    return result


def select_hardware(candidates, requirements):
    eligible = [
        item for item in candidates
        if item["memory_gb"] >= requirements["memory_gb_min"]
        and item["memory_bandwidth_gbps"] >= requirements["memory_bandwidth_gbps_min"]
        and item["ai_tops"] >= requirements["ai_tops_min"]
        and item["module_power_w_max"] <= requirements["module_power_w_max"]
    ]
    if not eligible:
        raise ValueError("no hardware candidate meets the frozen deployment requirements")
    return min(
        eligible,
        key=lambda item: (item["memory_gb"], item["ai_tops"], item["module_power_w_max"]),
    )


def profile_gate(profile, acceptance):
    return bool(
        profile["semantic_provider"] == "CUDAExecutionProvider"
        and profile["xfeat_provider"] == "CUDAExecutionProvider"
        and profile["control_pipeline"]["p95_ms"] <= acceptance["control_pipeline_p95_ms_max"]
        and profile["xfeat"]["p95_ms"] <= acceptance["xfeat_p95_ms_max"]
        and profile["nmpc"]["p95_ms"] <= acceptance["nmpc_p95_ms_max"]
    )
