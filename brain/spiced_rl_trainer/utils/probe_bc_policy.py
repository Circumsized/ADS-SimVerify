import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np
import onnxruntime as ort


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
V_MAX = 0.80
KAPPA_MAX = 1.25
HEADING_RECOVERY_ENTER_DEG = 60.0
HEADING_RECOVERY_EXIT_DEG = 35.0
HEADING_RECOVERY_W_MAX = 0.90
HEADING_RECOVERY_KP = 0.85


def row_float(row, key, default=0.0):
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def bearing_deg(local_goal_x, local_goal_y):
    return math.degrees(math.atan2(local_goal_y, local_goal_x))


def load_rows(dataset_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "spice_run_*.csv"))):
        with open(path, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                local_goal_x = row_float(row, "local_goal_x")
                local_goal_y = row_float(row, "local_goal_y")
                local_goal_dist = row_float(row, "local_goal_dist")
                if local_goal_dist < 0.05:
                    continue
                rows.append(
                    {
                        "bearing": bearing_deg(local_goal_x, local_goal_y),
                        "dist": local_goal_dist,
                        "action_v": row_float(row, "action_v_norm"),
                        "action_kappa": row_float(row, "action_kappa_norm"),
                        "cmd_v": row_float(row, "cmd_v"),
                        "cmd_w": row_float(row, "cmd_w"),
                        "current_v": abs(row_float(row, "current_v")),
                    }
                )
    return rows


def print_dataset_bins(rows):
    print("=" * 96)
    print("Dataset bearing -> human action statistics")
    print("=" * 96)
    bins = [(-180, -135), (-135, -90), (-90, -45), (-45, 0), (0, 45), (45, 90), (90, 135), (135, 180)]
    for low, high in bins:
        subset = [row for row in rows if low <= row["bearing"] < high]
        if not subset:
            print(f"{low:>4}..{high:<4} n=0")
            continue
        action_v = np.array([row["action_v"] for row in subset])
        action_kappa = np.array([row["action_kappa"] for row in subset])
        dist = np.array([row["dist"] for row in subset])
        moving = np.array([row["current_v"] > 0.03 for row in subset])
        print(
            f"{low:>4}..{high:<4} n={len(subset):5d} "
            f"v_mean={action_v.mean():.3f} "
            f"k_mean={action_kappa.mean():+.3f} "
            f"k_abs={np.abs(action_kappa).mean():.3f} "
            f"k_q25/50/75={np.quantile(action_kappa, [0.25, 0.50, 0.75])} "
            f"dist={dist.mean():.2f} moving={moving.mean():.1%}"
        )


def print_model_grid(model_path, distance):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("=" * 96)
    print("ONNX repeated-frame policy probes")
    print(f"Deployment supervisor: enter={HEADING_RECOVERY_ENTER_DEG:.0f}deg exit={HEADING_RECOVERY_EXIT_DEG:.0f}deg")
    print("=" * 96)
    for degrees in [-170, -150, -120, -90, -45, 0, 45, 90, 120, 150, 170]:
        radians = math.radians(degrees)
        local_goal_x = math.cos(radians) * distance
        local_goal_y = math.sin(radians) * distance
        frame = [local_goal_x * 0.20, local_goal_y * 0.20, distance * 0.20]
        state = np.array([frame * 5], dtype=np.float32)
        action = session.run(None, {input_name: state})[0][0]
        if abs(degrees) > HEADING_RECOVERY_ENTER_DEG:
            mode = "heading_recovery"
            v_des = 0.0
            w_ref = float(np.clip(HEADING_RECOVERY_KP * math.radians(degrees), -HEADING_RECOVERY_W_MAX, HEADING_RECOVERY_W_MAX))
        else:
            mode = "bc"
            v_des = float(np.clip(action[0], 0.0, 1.0) * V_MAX)
            kappa = float(np.clip(action[1], -1.0, 1.0) * KAPPA_MAX)
            w_ref = v_des * kappa
        print(
            f"bearing={degrees:+4d}deg "
            f"local_goal=({local_goal_x:+.2f},{local_goal_y:+.2f}) "
            f"mode={mode} "
            f"action=({action[0]:.3f},{action[1]:+.3f}) "
            f"cmd=({v_des:.3f},{w_ref:+.3f})"
        )


def main():
    parser = argparse.ArgumentParser(description="Probe Spice BC policy behavior against dataset bearing bins.")
    parser.add_argument("--dataset-dir", default=os.path.join(REPO_ROOT, "dataset", "purified"))
    parser.add_argument("--model", default=os.path.join(REPO_ROOT, "model", "spiced_brain.onnx"))
    parser.add_argument("--distance", type=float, default=6.0)
    args = parser.parse_args()

    rows = load_rows(args.dataset_dir)
    print(f"Loaded {len(rows)} non-terminal rows from {args.dataset_dir}")
    print_dataset_bins(rows)
    print_model_grid(args.model, args.distance)


if __name__ == "__main__":
    main()
