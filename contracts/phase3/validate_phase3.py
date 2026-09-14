#!/usr/bin/env python3
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE3_DIR = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_text):
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def metric(summary, name, statistic="mean"):
    return float(summary["metrics"][name][statistic])


def main():
    manifest_path = PHASE3_DIR / "domain_scene_baseline.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []

    decision = manifest["resolved_decision"]
    if decision["id"] != "P1-U001/G0" or decision["status"] != "closed":
        errors.append("P1-U001/G0 was not explicitly closed")
    if decision["deployment_domain_id"] != "indoor_warehouse_closed_track":
        errors.append("unexpected deployment domain")

    for role, scene in manifest["scene_roles"].items():
        path = resolve(scene["local_path"])
        if not path.is_file():
            errors.append(f"{role} scene is missing: {path}")
            continue
        actual = sha256(path)
        if actual != scene["sha256"]:
            errors.append(f"{role} scene hash mismatch: {actual}")

    primary = manifest["scene_roles"]["primary"]
    if primary["semantic_ready"] is not False:
        errors.append("primary vendor scene must remain explicitly semantic-not-ready")
    if primary["inventory"]["semantic_label_binding_count"] != 0:
        errors.append("primary inventory semantic count no longer matches the frozen source")
    if "wrapper USD" not in primary["integration"]:
        errors.append("vendor scene must be integrated through a project wrapper USD")

    taxonomy = manifest["semantic_taxonomy"]
    channels = taxonomy["channels"]
    if taxonomy["id"] != "warehouse_nav14_v1" or len(channels) != 14:
        errors.append("warehouse_nav14_v1 must have exactly 14 channels")
    if [channel["id"] for channel in channels] != list(range(14)):
        errors.append("semantic channel ids must be contiguous 0..13")
    names = [channel["name"] for channel in channels]
    if len(names) != len(set(names)):
        errors.append("semantic channel names must be unique")
    free_ids = [channel["id"] for channel in channels if channel["occupancy"] == "free"]
    if free_ids != [0, 1] or channels[-1]["occupancy"] != "occupied":
        errors.append("only floor and floor marking may be encoded as free")

    bev = manifest["bev_contract"]
    phase1 = json.loads((ROOT / manifest["phase1_contract"]).read_text(encoding="utf-8"))
    frozen_bev = phase1["coordinate_frames"]["BEV"]
    if bev["shape"] != [*frozen_bev["shape"], 14]:
        errors.append("Phase 3 BEV shape is incompatible with Phase 1")
    if not math.isclose(bev["meters_per_cell"], frozen_bev["meters_per_cell"], rel_tol=0, abs_tol=1e-12):
        errors.append("Phase 3 BEV resolution is incompatible with Phase 1")
    if bev["ego_origin_cell"] != frozen_bev["origin_cell"]:
        errors.append("Phase 3 BEV origin is incompatible with Phase 1")

    evidence = manifest["phase2_evidence"]
    evidence_path = resolve(evidence["path"])
    if not evidence_path.is_file():
        errors.append(f"Phase 2 evidence is missing: {evidence_path}")
    else:
        if sha256(evidence_path) != evidence["sha256"]:
            errors.append("Phase 2 evidence hash mismatch")
        summary = json.loads(evidence_path.read_text(encoding="utf-8"))
        if summary["frames"] != evidence["frames"]:
            errors.append("Phase 2 frame count mismatch")
        observed = evidence["observed"]
        checks = {
            "exact_a_ratio": float(summary["exact_a_ratio"]),
            "exact_b_ratio": float(summary["exact_b_ratio"]),
            "b_valid_ratio": float(summary["b_valid_ratio"]),
            "c_valid_ratio": float(summary["c_valid_ratio"]),
            "c_latency_p95_ms": metric(summary, "c_latency_ms", "p95"),
            "roi_occupied_ratio_c_mean": metric(summary, "roi_occupied_ratio_c"),
            "bc_false_occupied_rate_mean": metric(summary, "bc_false_occupied_rate"),
            "bc_free_iou_mean": metric(summary, "bc_free_iou"),
        }
        for name, actual in checks.items():
            if not math.isclose(actual, float(observed[name]), rel_tol=0, abs_tol=1e-12):
                errors.append(f"Phase 2 observed metric drift: {name}={actual}")
        if checks["roi_occupied_ratio_c_mean"] < 0.99 or checks["bc_false_occupied_rate_mean"] < 0.95:
            errors.append("Phase 2 evidence no longer demonstrates the frozen PIDNet collapse")

    gate = manifest["phase4_perception_gate"]
    if gate["status"] != "closed" or len(gate["blockers"]) < 3:
        errors.append("Phase 4 perception gate must remain closed with explicit blockers")

    phase1_sums = ROOT / "contracts/v1/SHA256SUMS"
    for line in phase1_sums.read_text(encoding="ascii").splitlines():
        expected, filename = line.split(maxsplit=1)
        candidate = phase1_sums.parent / filename
        if sha256(candidate) != expected:
            errors.append(f"Phase 1 frozen file drifted: {filename}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Phase 3 domain/scene baseline validation OK")
    print("G0 closed: indoor_warehouse_closed_track")
    print("Primary scene frozen: isaac6_warehouse_multiple_shelves")
    print("Phase 4 perception gate remains closed: semantic overlay, camera distortion and model adaptation are pending.")


if __name__ == "__main__":
    main()
