#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "contracts/phase3/domain_scene_baseline.json"
OVERLAY = ROOT / "assets/phase4/warehouse_nav14_overlay.usda"
OVERLAY_MANIFEST = ROOT / "assets/phase4/warehouse_nav14_overlay.manifest.json"
STATUS = ROOT / "contracts/phase4/phase4_status.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    phase3 = json.loads(PHASE3.read_text(encoding="utf-8"))
    overlay = json.loads(OVERLAY_MANIFEST.read_text(encoding="utf-8"))
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    errors = []

    if status["status"] != "p4_ab_frozen_p4c_closed":
        errors.append("Phase 4 status must distinguish P4-A/B from the closed P4-C gate")
    if sha256(ROOT / status["phase3_manifest"]["path"]) != status["phase3_manifest"]["sha256"]:
        errors.append("Phase 4 references a different Phase 3 baseline")
    p4a = status["p4_a_semantic_overlay"]
    if sha256(ROOT / p4a["overlay_path"]) != p4a["overlay_sha256"]:
        errors.append("frozen semantic overlay hash mismatch")
    if sha256(ROOT / p4a["manifest_path"]) != p4a["manifest_sha256"]:
        errors.append("frozen semantic overlay manifest hash mismatch")
    p4b = status["p4_b_data_loop"]
    frozen_capture = ROOT / p4b["capture_path"]
    if sha256(frozen_capture / "summary.json") != p4b["summary_sha256"]:
        errors.append("frozen Phase 4 capture summary hash mismatch")
    if sha256(frozen_capture / "evidence.png") != p4b["evidence_sha256"]:
        errors.append("frozen Phase 4 evidence hash mismatch")
    if p4b["training_dataset_ready"] is not False:
        errors.append("smoke evidence must not be marked training-ready")
    if status["p4_c_candidate_perception_gate"]["status"] != "closed":
        errors.append("candidate perception gate was opened without 1000-frame evidence")

    vendor = Path(overlay["vendor_usd"])
    frozen_scene = phase3["scene_roles"]["primary"]
    if overlay["taxonomy_id"] != phase3["semantic_taxonomy"]["id"]:
        errors.append("overlay taxonomy differs from Phase 3")
    if overlay["vendor_sha256_before"] != frozen_scene["sha256"]:
        errors.append("overlay was not built from the frozen vendor scene")
    if not vendor.is_file() or sha256(vendor) != frozen_scene["sha256"]:
        errors.append("vendor scene is missing or was modified")
    if overlay["mesh_count"] != frozen_scene["inventory"]["mesh_count"]:
        errors.append("overlay mesh count differs from frozen inventory")
    if overlay["labeled_mesh_count"] != overlay["mesh_count"]:
        errors.append("not every mesh is labeled")
    if overlay.get("stage_metadata") != {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "default_prim": "/Root",
    }:
        errors.append("overlay stage metadata differs from the frozen vendor scene")
    if sum(overlay["class_mesh_counts"].values()) != overlay["mesh_count"]:
        errors.append("semantic class counts do not cover every mesh exactly once")
    if set(overlay["class_mesh_counts"]) != {
        item["name"] for item in phase3["semantic_taxonomy"]["channels"]
    }:
        errors.append("overlay classes differ from warehouse_nav14_v1")
    if not OVERLAY.is_file():
        errors.append("overlay USD is missing")

    if args.capture:
        summary_path = args.capture / "summary.json"
        if not summary_path.is_file():
            errors.append(f"capture summary is missing: {summary_path}")
        else:
            capture = json.loads(summary_path.read_text(encoding="utf-8"))
            if capture["taxonomy_id"] != overlay["taxonomy_id"]:
                errors.append("capture taxonomy differs from overlay")
            if capture["frame_count"] != len(capture["frames"]):
                errors.append("capture frame count is inconsistent")
            if capture["shapes"]["semantic_gt_ipm"] != [14, 192, 192]:
                errors.append("capture BEV shape differs from frozen transport")
            if capture["overlay_manifest_sha256"] != sha256(OVERLAY_MANIFEST):
                errors.append("capture references a different overlay manifest")
            if capture["overlay_usd_sha256"] != sha256(OVERLAY):
                errors.append("capture references a different overlay USD")
            evidence = args.capture / capture["evidence"]
            if not evidence.is_file() or sha256(evidence) != capture["evidence_sha256"]:
                errors.append("capture evidence image is missing or has drifted")
            for frame in capture["frames"]:
                for field in ("archive", "jpeg"):
                    path = args.capture / frame[field]
                    if not path.is_file():
                        errors.append(f"capture file is missing: {path.name}")
                    elif sha256(path) != frame[f"{field}_sha256"]:
                        errors.append(f"capture hash mismatch: {path.name}")
                pixels = frame["semantic_class_pixel_counts"]
                total_pixels = sum(pixels.values())
                free_pixels = pixels["traversable_floor"] + pixels["floor_marking"]
                if total_pixels != 640 * 480:
                    errors.append(f"semantic pixel count is inconsistent: frame {frame['source_frame_id']}")
                if free_pixels == 0:
                    errors.append(f"semantic frame has no free-space evidence: frame {frame['source_frame_id']}")
                if pixels["unknown_or_unlabeled"] / total_pixels >= 0.99:
                    errors.append(f"semantic frame collapsed to unknown: frame {frame['source_frame_id']}")
            if not 0.01 < capture["ipm_valid_ratio"] < 0.99:
                errors.append("IPM valid ROI is empty or covers the entire BEV unexpectedly")
            ratios = capture["depth_roi_occupied_ratio"]
            if ratios["min"] >= 0.99:
                errors.append("smoke capture is all occupied within the depth-observed ROI")
            if ratios["max"] <= 0.01:
                errors.append("depth-lift smoke capture contains no obstacle evidence")
            for frame in capture["frames"]:
                if frame["depth_finite_ratio"] < 0.99:
                    errors.append(f"metric depth is incomplete: frame {frame['source_frame_id']}")
                if not 0.05 < frame["depth_p50_m"] < 20.0:
                    errors.append(f"metric depth scale is invalid: frame {frame['source_frame_id']}")
            if capture["candidate_perception"] is not None:
                errors.append("smoke evidence must not impersonate a candidate model gate")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Phase 4 semantic overlay validation OK")
    print(f"Labeled meshes: {overlay['labeled_mesh_count']}/{overlay['mesh_count']}")
    if args.capture:
        print(f"Phase 4 synchronized smoke capture validation OK: {args.capture}")
    print("Candidate perception gate remains closed until a warehouse model passes 1000 synchronized frames.")


if __name__ == "__main__":
    main()
