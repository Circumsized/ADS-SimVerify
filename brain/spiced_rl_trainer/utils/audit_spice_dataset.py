import argparse
import csv
import glob
import json
import math
import os
from statistics import mean


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "dataset")

V_MAX = 0.80
KAPPA_MAX = 1.25

REQUIRED_FIELDS = [
    "timestamp",
    "odom_x",
    "odom_y",
    "odom_yaw",
    "local_goal_x",
    "local_goal_y",
    "local_goal_dist",
    "current_v",
    "action_v_norm",
    "action_kappa_norm",
]
OPTIONAL_FIELDS = ["cmd_v", "cmd_w"]

REACH_DIST_M = 0.40
MOVING_SPEED_MPS = 0.03
BC_SPEED_MPS = 0.05
IDLE_CMD_EPS = 0.02
IDLE_SPEED_EPS = 0.03
MIN_RAW_ROWS = 300
MIN_BC_SPAN_ROWS = 150
MIN_MAX_SPEED_MPS = 0.20
MAX_FRAME_JUMP_M = 0.75


def row_has_value(row, key):
    return row.get(key) not in (None, "")


def row_float(row, key, default=0.0):
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def finite_values(values):
    return all(math.isfinite(value) for value in values)


def pct(value):
    return f"{value * 100.0:.1f}%"


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def compass_bin_from_xy(x, y):
    degrees = math.degrees(math.atan2(y, x))
    if degrees < 0.0:
        degrees += 360.0
    labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((degrees + 22.5) % 360.0) // 45.0)
    return labels[idx]


def world_goal_delta(local_goal_x, local_goal_y, odom_yaw):
    dx = local_goal_x * math.cos(odom_yaw) - local_goal_y * math.sin(odom_yaw)
    dy = local_goal_x * math.sin(odom_yaw) + local_goal_y * math.cos(odom_yaw)
    return dx, dy


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def bc_trim_window(rows):
    if len(rows) < MIN_RAW_ROWS:
        return None
    max_speed = max(abs(row_float(row, "current_v", row_float(row, "cmd_v", 0.0))) for row in rows)
    if max_speed < MIN_MAX_SPEED_MPS:
        return None
    start_idx = 0
    for idx, row in enumerate(rows):
        speed = abs(row_float(row, "current_v", row_float(row, "cmd_v", 0.0)))
        if speed > BC_SPEED_MPS:
            start_idx = idx
            break
    end_idx = len(rows) - 1
    for idx in range(len(rows) - 1, -1, -1):
        row = rows[idx]
        speed = abs(row_float(row, "current_v", row_float(row, "cmd_v", 0.0)))
        if speed > BC_SPEED_MPS or row_float(row, "local_goal_dist") > 0.20:
            end_idx = idx
            break
    if end_idx - start_idx < MIN_BC_SPAN_ROWS:
        return None
    return start_idx, end_idx


def analyze_file(path):
    fieldnames, rows = read_csv(path)
    missing_required = [field for field in REQUIRED_FIELDS if field not in fieldnames]
    missing_optional = [field for field in OPTIONAL_FIELDS if field not in fieldnames]

    metrics = {
        "file": os.path.basename(path),
        "path": path,
        "bytes": os.path.getsize(path),
        "rows": len(rows),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "status": "reject",
        "issues": [],
        "recommendations": [],
    }

    if missing_required:
        metrics["issues"].append("Required training fields are missing")
        metrics["recommendations"].append("Not suitable for training; repair fields or discard this segment")
        return metrics
    if not rows:
        metrics["issues"].append("Empty file or header only")
        metrics["recommendations"].append("Discard")
        return metrics

    ts = [row_float(row, "timestamp") for row in rows]
    xs = [row_float(row, "odom_x") for row in rows]
    ys = [row_float(row, "odom_y") for row in rows]
    yaws = [row_float(row, "odom_yaw") for row in rows]
    goal_x = [row_float(row, "local_goal_x") for row in rows]
    goal_y = [row_float(row, "local_goal_y") for row in rows]
    goal_dist = [row_float(row, "local_goal_dist") for row in rows]
    current_v = [abs(row_float(row, "current_v")) for row in rows]
    action_v = [row_float(row, "action_v_norm") for row in rows]
    action_kappa = [row_float(row, "action_kappa_norm") for row in rows]
    cmd_v = [row_float(row, "cmd_v", action_v[idx] * V_MAX) for idx, row in enumerate(rows)]
    cmd_w = [row_float(row, "cmd_w", 0.0) for row in rows]

    numeric_values = ts + xs + ys + yaws + goal_x + goal_y + goal_dist + current_v + action_v + action_kappa + cmd_v + cmd_w
    metrics["finite"] = finite_values(numeric_values)
    if not metrics["finite"]:
        metrics["issues"].append("NaN/Inf or unparseable numeric values present")

    jumps = [math.hypot(xs[idx] - xs[idx - 1], ys[idx] - ys[idx - 1]) for idx in range(1, len(rows))]
    path_len = sum(jumps)
    direct_span = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    bbox_area = max(xs) - min(xs)
    bbox_area *= max(ys) - min(ys)
    duration_s = max(0.0, ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    hz = (len(rows) - 1) / duration_s if duration_s > 0.0 else 0.0
    idle_mask = [
        abs(cmd_v[idx]) < IDLE_CMD_EPS
        and abs(cmd_w[idx]) < IDLE_CMD_EPS
        and current_v[idx] < IDLE_SPEED_EPS
        for idx in range(len(rows))
    ]
    moving_ratio = sum(v > MOVING_SPEED_MPS for v in current_v) / len(rows)
    idle_ratio = sum(idle_mask) / len(rows)
    cmd_nonzero_ratio = sum(abs(cmd_v[idx]) > IDLE_CMD_EPS or abs(cmd_w[idx]) > IDLE_CMD_EPS for idx in range(len(rows))) / len(rows)
    v_sat_ratio = sum(abs(value) > 0.98 for value in action_v) / len(rows)
    kappa_sat_ratio = sum(abs(value) > 0.98 for value in action_kappa) / len(rows)
    reverse_cmd_ratio = sum(value < -0.01 for value in cmd_v) / len(rows)
    reached = min(goal_dist) <= REACH_DIST_M
    progress_ratio = 1.0 - (goal_dist[-1] / max(goal_dist[0], 1e-6))
    trim_window = bc_trim_window(rows)
    bc_samples = 0 if trim_window is None else max(0, trim_window[1] - trim_window[0] + 1 - 4)

    sorted_speed = sorted(current_v)
    sorted_kappa = sorted(abs(value) for value in action_kappa)
    world_dx0, world_dy0 = world_goal_delta(goal_x[0], goal_y[0], yaws[0])

    metrics.update(
        {
            "duration_s": duration_s,
            "hz": hz,
            "goal_bin": compass_bin_from_xy(goal_x[0], goal_y[0]),
            "local_goal_bin": compass_bin_from_xy(goal_x[0], goal_y[0]),
            "world_goal_bin": compass_bin_from_xy(world_dx0, world_dy0),
            "goal_dist_first": goal_dist[0],
            "goal_dist_min": min(goal_dist),
            "goal_dist_last": goal_dist[-1],
            "reached": reached,
            "progress_ratio": progress_ratio,
            "path_len_m": path_len,
            "direct_span_m": direct_span,
            "tortuosity": path_len / max(direct_span, 1e-6),
            "bbox_area_m2": bbox_area,
            "max_frame_jump_m": max(jumps) if jumps else 0.0,
            "moving_ratio": moving_ratio,
            "idle_ratio": idle_ratio,
            "cmd_nonzero_ratio": cmd_nonzero_ratio,
            "v_sat_ratio": v_sat_ratio,
            "kappa_sat_ratio": kappa_sat_ratio,
            "reverse_cmd_ratio": reverse_cmd_ratio,
            "speed_p50": quantile(sorted_speed, 0.50),
            "speed_p95": quantile(sorted_speed, 0.95),
            "kappa_abs_p50": quantile(sorted_kappa, 0.50),
            "kappa_abs_p95": quantile(sorted_kappa, 0.95),
            "bc_trim_start": None if trim_window is None else trim_window[0],
            "bc_trim_end": None if trim_window is None else trim_window[1],
            "bc_samples_est": bc_samples,
        }
    )

    if len(rows) < 50:
        metrics["issues"].append("Too few rows")
    elif len(rows) < MIN_RAW_ROWS:
        metrics["issues"].append("Row count below the BC training threshold")
    if idle_ratio > 0.50:
        metrics["issues"].append("Too many idle/waiting samples")
        metrics["recommendations"].append("Filter all-zero idle samples during post-processing")
    if path_len < 0.50:
        metrics["issues"].append("Trajectory shows almost no movement")
    if not reached:
        metrics["issues"].append("Goal threshold not reached")
        metrics["recommendations"].append("Isolate incomplete segments unless training recovery behavior")
    if metrics["max_frame_jump_m"] > MAX_FRAME_JUMP_M:
        metrics["issues"].append("Suspected teleport or towing jump detected")
        metrics["recommendations"].append("Split at jumps or discard nearby windows")
    if v_sat_ratio > 0.60:
        metrics["issues"].append("Speed action is saturated for too long")
        metrics["recommendations"].append("Collect more low-speed obstacle-avoidance samples")
    if kappa_sat_ratio > 0.45:
        metrics["issues"].append("Curvature action is saturated for too long")
        metrics["recommendations"].append("Collect more smooth turns and low-curvature corrections")
    if reverse_cmd_ratio > 0.0:
        metrics["issues"].append("Negative speed commands violate the Sim2Real-AD PAM non-negative speed contract")
    if trim_window is None:
        metrics["issues"].append("BC training-window filtering failed")
    if progress_ratio < 0.50:
        metrics["issues"].append("Insufficient goal-distance convergence")
    if moving_ratio < 0.25:
        metrics["issues"].append("Low effective-motion ratio")

    if not metrics["recommendations"]:
        metrics["recommendations"].append("Keep; proceed to hindsight/purified processing")

    has_jump = metrics["max_frame_jump_m"] > MAX_FRAME_JUMP_M
    if not metrics["finite"] or missing_required or path_len < 0.50 or idle_ratio > 0.90:
        metrics["status"] = "reject"
    elif trim_window is None or not reached or moving_ratio < 0.25 or has_jump:
        metrics["status"] = "review"
    else:
        metrics["status"] = "keep"
    return metrics


def aggregate_report(file_metrics):
    nonempty = [item for item in file_metrics if item["rows"] > 0]
    keep = [item for item in file_metrics if item["status"] == "keep"]
    review = [item for item in file_metrics if item["status"] == "review"]
    reject = [item for item in file_metrics if item["status"] == "reject"]
    local_bins = {}
    world_bins = {}
    for item in nonempty:
        if "local_goal_bin" in item:
            local_bins[item["local_goal_bin"]] = local_bins.get(item["local_goal_bin"], 0) + 1
        if "world_goal_bin" in item:
            world_bins[item["world_goal_bin"]] = world_bins.get(item["world_goal_bin"], 0) + 1
    return {
        "files_total": len(file_metrics),
        "files_nonempty": len(nonempty),
        "keep": len(keep),
        "review": len(review),
        "reject": len(reject),
        "rows_total": sum(item["rows"] for item in file_metrics),
        "rows_keep": sum(item["rows"] for item in keep),
        "bc_samples_est": sum(item.get("bc_samples_est", 0) for item in keep),
        "goal_bins": local_bins,
        "local_goal_bins": local_bins,
        "world_goal_bins": world_bins,
        "mean_moving_ratio_keep": mean([item["moving_ratio"] for item in keep]) if keep else 0.0,
        "mean_idle_ratio_keep": mean([item["idle_ratio"] for item in keep]) if keep else 0.0,
    }


def print_report(file_metrics, aggregate):
    print("=" * 100)
    print("Spice dataset quality audit")
    print("=" * 100)
    print(
        f"Files: {aggregate['files_total']} | non-empty: {aggregate['files_nonempty']} | "
        f"keep: {aggregate['keep']} | review: {aggregate['review']} | reject: {aggregate['reject']}"
    )
    print(
        f"total rows: {aggregate['rows_total']} | kept rows: {aggregate['rows_keep']} | "
        f"estimated BC samples: {aggregate['bc_samples_est']}"
    )
    print(
        f"kept-segment mean moving ratio: {pct(aggregate['mean_moving_ratio_keep'])} | "
        f"kept-segment mean idle ratio: {pct(aggregate['mean_idle_ratio_keep'])}"
    )
    print(f"Local goal directions: {aggregate['local_goal_bins']}")
    print(f"World goal directions: {aggregate['world_goal_bins']}")
    print("-" * 100)
    header = (
        f"{'file':<34} {'status':<7} {'rows':>6} {'dist':>17} {'move':>7} "
        f"{'idle':>7} {'v_sat':>7} {'k_sat':>7} {'path':>7} {'bc':>7}"
    )
    print(header)
    print("-" * 100)
    for item in file_metrics:
        if item["rows"] == 0:
            dist_text = "-"
            move = idle = v_sat = k_sat = "-"
            path_len = "-"
            bc_samples = "0"
        else:
            dist_text = f"{item.get('goal_dist_first', 0.0):.1f}->{item.get('goal_dist_min', 0.0):.1f}"
            move = pct(item.get("moving_ratio", 0.0))
            idle = pct(item.get("idle_ratio", 0.0))
            v_sat = pct(item.get("v_sat_ratio", 0.0))
            k_sat = pct(item.get("kappa_sat_ratio", 0.0))
            path_len = f"{item.get('path_len_m', 0.0):.1f}"
            bc_samples = str(item.get("bc_samples_est", 0))
        print(
            f"{item['file']:<34} {item['status']:<7} {item['rows']:>6} {dist_text:>17} "
            f"{move:>7} {idle:>7} {v_sat:>7} {k_sat:>7} {path_len:>7} {bc_samples:>7}"
        )
        if item["issues"]:
            print("  Issues: " + "；".join(item["issues"]))
        if item["recommendations"]:
            print("  Recommendations: " + "；".join(dict.fromkeys(item["recommendations"])))
    print("-" * 100)
    if aggregate["reject"] or aggregate["review"]:
        print("Conclusion: post-processing required; filter or isolate reject segments and review review segments.")
    else:
        print("Conclusion: data can proceed to hindsight/purified processing and BC training.")
    print("Pipeline: raw capture -> audit -> isolate reject -> hindsight_purifier -> train_bc_anchor")
    print("=" * 100)


def write_csv(path, file_metrics):
    fieldnames = [
        "file",
        "status",
        "rows",
        "goal_bin",
        "local_goal_bin",
        "world_goal_bin",
        "goal_dist_first",
        "goal_dist_min",
        "goal_dist_last",
        "moving_ratio",
        "idle_ratio",
        "v_sat_ratio",
        "kappa_sat_ratio",
        "path_len_m",
        "max_frame_jump_m",
        "bc_samples_est",
        "issues",
        "recommendations",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in file_metrics:
            row = dict(item)
            row["issues"] = "；".join(item.get("issues", []))
            row["recommendations"] = "；".join(item.get("recommendations", []))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Audit Spice expert CSV dataset quality.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--pattern", default="spice_run_*.csv")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.dataset_dir, args.pattern)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched {os.path.join(args.dataset_dir, args.pattern)}")

    file_metrics = [analyze_file(path) for path in csv_paths]
    aggregate = aggregate_report(file_metrics)
    print_report(file_metrics, aggregate)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as file:
            json.dump({"aggregate": aggregate, "files": file_metrics}, file, ensure_ascii=False, indent=2)
    if args.csv_out:
        write_csv(args.csv_out, file_metrics)


if __name__ == "__main__":
    main()
