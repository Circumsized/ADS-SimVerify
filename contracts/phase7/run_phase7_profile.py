#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import psutil


ROOT = Path(__file__).resolve().parents[2]
PHASE5 = ROOT / "contracts/phase5"
PHASE7 = ROOT / "contracts/phase7"
sys.path.insert(0, str(PHASE5))
sys.path.insert(0, str(PHASE7))

from oracle_nmpc_closed_loop import OracleGrid, load_solver  # noqa: E402
from phase5b_shadow_replay import load_trajectory, occupancy_from_semantic  # noqa: E402
from phase5c3_candidate_shadow import candidate_depth_lift  # noqa: E402
from phase5f_dual_nmpc_shadow import (  # noqa: E402
    bev_obstacle_parameters,
    scenario_paths,
    solve_shadow,
)
from phase7_profile import (  # noqa: E402
    profile_gate,
    select_hardware,
    stream_bandwidth,
    timing_summary,
)


def prepare_semantic_tensor(decoded_bgr):
    image = cv2.resize(decoded_bgr, (320, 240), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(np.transpose((image - mean) / std, (2, 0, 1))[None])


def prepare_xfeat_tensor(decoded_bgr):
    gray = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(640.0 / width, 640.0 / height)
    resized = cv2.resize(
        gray,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((640, 640), dtype=np.float32)
    top = (640 - resized.shape[0]) // 2
    left = (640 - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    mean = float(canvas.mean())
    std = max(float(canvas.std()), 1e-6)
    return np.ascontiguousarray(((canvas - mean) / std)[None, None], dtype=np.float32)


def summarize_nvidia_samples(lines, baseline_vram_mib):
    parsed = []
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 5:
            continue
        try:
            parsed.append(tuple(float(value) for value in fields[-4:]))
        except ValueError:
            continue
    if not parsed:
        raise ValueError("nvidia-smi produced no parseable samples")
    values = np.asarray(parsed, dtype=np.float64)
    return {
        "samples": int(values.shape[0]),
        "vram_peak_mib": float(values[:, 0].max()),
        "vram_incremental_peak_mib": float(max(0.0, values[:, 0].max() - baseline_vram_mib)),
        "gpu_utilization_mean_percent": float(values[:, 1].mean()),
        "gpu_utilization_peak_percent": float(values[:, 1].max()),
        "gpu_power_mean_w": float(values[:, 2].mean()),
        "gpu_power_peak_w": float(values[:, 2].max()),
        "gpu_temperature_peak_c": float(values[:, 3].max()),
    }


def decode_semantic(logits, target_shape):
    logits = np.asarray(logits)[0]
    best = np.argmax(logits, axis=0).astype(np.uint8)
    maximum = np.max(logits, axis=0, keepdims=True)
    confidence = 1.0 / np.exp(logits - maximum).sum(axis=0)
    best[confidence < 0.50] = 13
    return cv2.resize(best, target_shape, interpolation=cv2.INTER_NEAREST)


def create_session(model, providers):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(model), sess_options=options, providers=providers)


def nvidia_value(query):
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.splitlines()[0].strip())


def start_nvidia_sampler():
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            "100",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_nvidia_sampler(process):
    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    if not stdout.strip():
        raise RuntimeError(f"nvidia-smi sampler returned no data: {stderr.strip()}")
    return stdout.splitlines()


def load_fixture(path, count):
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    frames = manifest["frames"][:count]
    return [
        {
            "id": int(frame["source_frame_id"]),
            "jpeg": np.frombuffer((path / frame["jpeg"]).read_bytes(), dtype=np.uint8),
            "depth": np.load(path / frame["depth"]).astype(np.float32),
        }
        for frame in frames
    ]


def measure(session, tensor, iterations):
    values = []
    input_name = session.get_inputs()[0].name
    for _ in range(iterations):
        started = time.perf_counter_ns()
        session.run(None, {input_name: tensor})
        values.append((time.perf_counter_ns() - started) / 1e6)
    return timing_summary(values)


def inherited_runtime_evidence():
    phase5d = json.loads((ROOT / "contracts/phase5/phase5d_status.json").read_text())
    phase5i = json.loads((ROOT / "contracts/phase5/phase5i_status.json").read_text())
    phase5k = json.loads((ROOT / "contracts/phase5/phase5k_status.json").read_text())
    phase6 = json.loads((ROOT / "contracts/phase6/phase6_status.json").read_text())
    return {
        "phase5d_dora_perception_p95_ms": phase5d["runtime"]["latency_p95_ms"],
        "phase5i_dora_sensor_to_wheel_p95_ms": phase5i["aggregate"]["maximum_sensor_to_wheel_p95_ms"],
        "phase5k_hour_frames": phase5k["hour_endurance"]["frames"],
        "phase5k_maximum_fault_recovery_frames": phase5k["fault_evidence"]["maximum_recoverable_fault_stop_latency_frames"],
        "phase6_sensor_to_wheel_p95_ms": phase6["aggregate"]["sensor_to_wheel_p95_ms_max"],
    }


def run_profile(contract):
    benchmark = contract["benchmark"]
    fixture = load_fixture(ROOT / benchmark["fixture"], benchmark["control_iterations"])
    decoded = cv2.imdecode(fixture[0]["jpeg"], cv2.IMREAD_COLOR)
    semantic_tensor = prepare_semantic_tensor(decoded)
    xfeat_tensor = prepare_xfeat_tensor(decoded)

    baseline_vram = nvidia_value("memory.used")
    sampler = start_nvidia_sampler()
    process = psutil.Process()
    rss_baseline = process.memory_info().rss / 2**20
    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()
    try:
        providers = [benchmark["preferred_provider"], benchmark["fallback_provider"]]
        semantic = create_session(ROOT / benchmark["semantic_model"], providers)
        xfeat = create_session(ROOT / benchmark["xfeat_model"], providers)
        semantic_cpu = create_session(ROOT / benchmark["semantic_model"], [benchmark["fallback_provider"]])
        xfeat_cpu = create_session(ROOT / benchmark["xfeat_model"], [benchmark["fallback_provider"]])

        for _ in range(benchmark["warmup_iterations"]):
            semantic.run(None, {semantic.get_inputs()[0].name: semantic_tensor})
            xfeat.run(None, {xfeat.get_inputs()[0].name: xfeat_tensor})

        phase3 = json.loads((ROOT / "contracts/phase3/domain_scene_baseline.json").read_text())
        phase5a = json.loads((ROOT / "contracts/phase5/phase5a_status.json").read_text())
        _, trajectory = load_trajectory(phase5a)
        trajectory = trajectory[: len(fixture)]
        manifest_path = ROOT / phase5a["oracle_map"]["manifest"]["path"]
        manifest = json.loads(manifest_path.read_text())
        grid = OracleGrid(manifest_path.parent / manifest["archive"], manifest)
        paths = scenario_paths(grid)
        solver = load_solver()
        sensor, bev = phase3["sensor_geometry"], phase3["bev_contract"]

        stages = {name: [] for name in (
            "jpeg_decode", "semantic_preprocess", "semantic_inference",
            "depth_lift_bev", "nmpc", "control_pipeline",
        )}
        rss_samples = []
        semantic_input = semantic.get_inputs()[0].name
        for fixture_frame, trajectory_frame in zip(fixture, trajectory):
            total_started = time.perf_counter_ns()
            started = time.perf_counter_ns()
            image = cv2.imdecode(fixture_frame["jpeg"], cv2.IMREAD_COLOR)
            stages["jpeg_decode"].append((time.perf_counter_ns() - started) / 1e6)

            started = time.perf_counter_ns()
            tensor = prepare_semantic_tensor(image)
            stages["semantic_preprocess"].append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            logits = semantic.run(None, {semantic_input: tensor})[0]
            classes = decode_semantic(logits, (image.shape[1], image.shape[0]))
            stages["semantic_inference"].append((time.perf_counter_ns() - started) / 1e6)

            started = time.perf_counter_ns()
            semantic_bev, observed = candidate_depth_lift(
                classes, fixture_frame["depth"], sensor, bev
            )
            occupied = occupancy_from_semantic(semantic_bev)
            obstacles, _ = bev_obstacle_parameters(occupied, observed, bev)
            stages["depth_lift_bev"].append((time.perf_counter_ns() - started) / 1e6)

            scenario, path = paths[trajectory_frame["scenario"]]
            state = np.asarray([
                trajectory_frame["x_m"], trajectory_frame["y_m"],
                trajectory_frame["yaw_rad"], trajectory_frame["velocity_mps"],
            ], dtype=np.float64)
            _, status, solve_ms, _, _, _ = solve_shadow(
                solver, path, state, scenario.goal, obstacles
            )
            if status != 0:
                raise RuntimeError(f"NMPC failed on fixture frame {fixture_frame['id']}: {status}")
            stages["nmpc"].append(solve_ms)
            stages["control_pipeline"].append((time.perf_counter_ns() - total_started) / 1e6)
            rss_samples.append(process.memory_info().rss / 2**20)

        xfeat_preprocess, xfeat_inference, xfeat_total = [], [], []
        xfeat_input = xfeat.get_inputs()[0].name
        for index in range(benchmark["xfeat_iterations"]):
            image = cv2.imdecode(fixture[index]["jpeg"], cv2.IMREAD_COLOR)
            total_started = time.perf_counter_ns()
            started = time.perf_counter_ns()
            tensor = prepare_xfeat_tensor(image)
            xfeat_preprocess.append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            xfeat.run(None, {xfeat_input: tensor})
            xfeat_inference.append((time.perf_counter_ns() - started) / 1e6)
            xfeat_total.append((time.perf_counter_ns() - total_started) / 1e6)
            rss_samples.append(process.memory_info().rss / 2**20)

        cpu_iterations = benchmark["cpu_fallback_iterations"]
        fallback = {
            "semantic_inference": measure(semantic_cpu, semantic_tensor, cpu_iterations),
            "xfeat_inference": measure(xfeat_cpu, xfeat_tensor, cpu_iterations),
        }
    finally:
        gpu_lines = stop_nvidia_sampler(sampler)

    wall_s = time.perf_counter() - wall_start
    cpu_end = process.cpu_times()
    cpu_s = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    timings = {name: timing_summary(values) for name, values in stages.items()}
    timings["xfeat_preprocess"] = timing_summary(xfeat_preprocess)
    timings["xfeat_inference"] = timing_summary(xfeat_inference)
    timings["xfeat_total"] = timing_summary(xfeat_total)
    selected = select_hardware(contract["hardware_candidates"], contract["recommended_requirements"])
    gate_input = {
        "semantic_provider": semantic.get_providers()[0],
        "xfeat_provider": xfeat.get_providers()[0],
        "control_pipeline": timings["control_pipeline"],
        "xfeat": timings["xfeat_total"],
        "nmpc": timings["nmpc"],
    }
    return {
        "schema_version": "phase7-deployment-profile-v1",
        "status": "deployment_profile_passed" if profile_gate(gate_input, contract["acceptance"]) else "deployment_profile_rejected",
        "host": {
            "hostname": platform.node(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpu_count": psutil.cpu_count(),
            "ram_total_gib": psutil.virtual_memory().total / 2**30,
            "onnxruntime_version": ort.__version__,
        },
        "providers": {
            "available": ort.get_available_providers(),
            "semantic": semantic.get_providers(), "xfeat": xfeat.get_providers(),
            "semantic_cpu_fallback": semantic_cpu.get_providers(),
            "xfeat_cpu_fallback": xfeat_cpu.get_providers(),
        },
        "benchmark": benchmark,
        "timings": timings,
        "cpu_fallback_timings": fallback,
        "resources": {
            "process_rss_baseline_mib": rss_baseline,
            "process_rss_peak_mib": max(rss_samples),
            "process_rss_incremental_peak_mib": max(rss_samples) - rss_baseline,
            "profile_wall_s": wall_s,
            "process_cpu_s": cpu_s,
            "process_cpu_core_equivalents": cpu_s / max(wall_s, 1e-9),
            "desktop_gpu": summarize_nvidia_samples(gpu_lines, baseline_vram),
        },
        "models": {
            "semantic": {"path": benchmark["semantic_model"], "bytes": (ROOT / benchmark["semantic_model"]).stat().st_size},
            "xfeat": {"path": benchmark["xfeat_model"], "bytes": (ROOT / benchmark["xfeat_model"]).stat().st_size},
            "active_total_bytes": sum((ROOT / benchmark[key]).stat().st_size for key in ("semantic_model", "xfeat_model")),
            "excluded": contract["scope"]["excluded_models"],
        },
        "declared_stream_bandwidth": stream_bandwidth(contract["stream_bandwidth"]),
        "inherited_runtime_evidence": inherited_runtime_evidence(),
        "recommendation": {
            "selected": selected,
            "requirements": contract["recommended_requirements"],
            "minimum_prototype": contract["hardware_candidates"][0],
            "high_headroom": contract["hardware_candidates"][-1],
            "target_hardware_benchmark_required": True,
        },
        "sensor_candidate": contract["sensor_candidate"],
        "limitations": [
            "x86 RTX 3060 latency is not a prediction of Jetson Orin latency",
            "XFeat profile includes ONNX inference and preprocessing but excludes Rust sparse postprocessing",
            "desktop GPU power is not Jetson module or complete system power",
            "real camera calibration and target-hardware rerun remain mandatory",
        ],
        "gate_passed": profile_gate(gate_input, contract["acceptance"]),
        "hardware_purchase_recommendation_allowed": profile_gate(gate_input, contract["acceptance"]),
        "real_vehicle_control_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts/phase7_profile" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((PHASE7 / "phase7_contract.json").read_text(encoding="utf-8"))
    summary = run_profile(contract)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"Phase 7 profile: {summary_path}")
    if not summary["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
