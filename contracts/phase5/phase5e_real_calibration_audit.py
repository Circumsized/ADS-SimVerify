#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts/phase5/phase5e_contract.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration(calibration, contract):
    errors = []
    if calibration.get("schema_version") != "phase5e-real-camera-calibration-v1":
        errors.append("real calibration schema_version is invalid")
    required_identity = contract["device_identity_required"]
    identity = calibration.get("device_identity", {})
    for key in required_identity:
        if not identity.get(key):
            errors.append(f"missing device_identity.{key}")
    if calibration.get("image_size") != contract["image_size"]:
        errors.append("calibration image size differs from the frozen contract")
    intrinsics = calibration.get("intrinsics", {})
    matrix = np.asarray(intrinsics.get("camera_matrix", []), dtype=np.float64)
    distortion = np.asarray(intrinsics.get("distortion", []), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        errors.append("camera_matrix must be a finite 3x3 matrix")
    if distortion.shape != (5,) or not np.isfinite(distortion).all():
        errors.append("distortion must contain finite k1,k2,p1,p2,k3")
    quality = intrinsics.get("quality", {})
    intrinsics_gate = contract["intrinsics"]
    if int(quality.get("checkerboard_views", 0)) < intrinsics_gate["minimum_checkerboard_views"]:
        errors.append("insufficient checkerboard views")
    if int(quality.get("occupied_image_bins_3x3", 0)) < intrinsics_gate["minimum_occupied_image_bins_3x3"]:
        errors.append("checkerboard image coverage is insufficient")
    if float(quality.get("rms_reprojection_error_px", math.inf)) > intrinsics_gate["rms_reprojection_error_px_max"]:
        errors.append("intrinsic RMS reprojection error exceeds the gate")
    if float(quality.get("per_view_p95_error_px", math.inf)) > intrinsics_gate["per_view_p95_error_px_max"]:
        errors.append("intrinsic per-view p95 error exceeds the gate")

    extrinsics = calibration.get("extrinsics", {})
    transform = np.asarray(extrinsics.get("T_body_camera", []), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        errors.append("T_body_camera must be a finite 4x4 matrix")
    else:
        rotation = transform[:3, :3]
        orthogonality = float(np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro"))
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3:
            errors.append("T_body_camera rotation determinant is not +1")
        if orthogonality > contract["extrinsics"]["rotation_orthogonality_error_max"]:
            errors.append("T_body_camera rotation is not orthonormal")
    if not extrinsics.get("target_based_measurement"):
        errors.append("target-based camera-to-body measurement evidence is missing")
    if float(extrinsics.get("translation_uncertainty_m", math.inf)) > contract["extrinsics"]["translation_uncertainty_m_max"]:
        errors.append("extrinsic translation uncertainty exceeds the gate")
    if float(extrinsics.get("rotation_uncertainty_deg", math.inf)) > contract["extrinsics"]["rotation_uncertainty_deg_max"]:
        errors.append("extrinsic rotation uncertainty exceeds the gate")

    depth = calibration.get("metric_depth", {})
    depth_gate = contract["metric_depth"]
    if not depth.get("registered_to_rgb"):
        errors.append("metric depth is not registered to RGB")
    if depth.get("units") != "meters":
        errors.append("metric depth units are not meters")
    if float(depth.get("plane_scale_relative_error", math.inf)) > depth_gate["plane_scale_relative_error_max"]:
        errors.append("metric depth plane-scale error exceeds the gate")
    if float(depth.get("paired_frame_ratio", 0.0)) < depth_gate["paired_frame_ratio_min"]:
        errors.append("RGB/depth paired-frame ratio is below the gate")
    for evidence in calibration.get("evidence", []):
        path = ROOT / evidence["path"]
        if not path.is_file() or sha256(path) != evidence["sha256"]:
            errors.append(f"calibration evidence missing or drifted: {evidence['path']}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts/phase5e_calibration" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))["real_camera_calibration"]
    devices = sorted(glob.glob("/dev/video*"))
    errors = []
    calibration_hash = None
    if not devices:
        errors.append("no /dev/video* real camera or depth device is present")
    if args.calibration is None:
        errors.append("no real camera calibration file was supplied")
    elif not args.calibration.is_file():
        errors.append(f"calibration file does not exist: {args.calibration}")
    else:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        calibration_hash = sha256(args.calibration)
        errors.extend(validate_calibration(calibration, contract))
    summary = {
        "schema_version": "phase5e-real-calibration-audit-v1",
        "status": "calibration_gate_passed" if not errors else "calibration_gate_blocked",
        "video_devices": devices,
        "calibration": str(args.calibration.resolve()) if args.calibration else None,
        "calibration_sha256": calibration_hash,
        "errors": errors,
        "gate_passed": not errors,
        "control_promotion_allowed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))
    print(f"Phase 5-E calibration audit: {output}")


if __name__ == "__main__":
    main()
