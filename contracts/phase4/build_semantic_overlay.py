#!/usr/bin/env python3
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE3_MANIFEST = ROOT / "contracts/phase3/domain_scene_baseline.json"
OUTPUT_DIR = ROOT / "assets/phase4"
OUTPUT_USD = OUTPUT_DIR / "warehouse_nav14_overlay.usda"
OUTPUT_MANIFEST = OUTPUT_DIR / "warehouse_nav14_overlay.manifest.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from warehouse_semantics import CHANNELS, TAXONOMY_ID, classify_prim_path  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    phase3 = json.loads(PHASE3_MANIFEST.read_text(encoding="utf-8"))
    vendor = Path(phase3["scene_roles"]["primary"]["local_path"])
    expected_vendor_hash = phase3["scene_roles"]["primary"]["sha256"]
    if sha256(vendor) != expected_vendor_hash:
        raise SystemExit("vendor USD hash differs from the frozen Phase 3 baseline")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd, UsdGeom
        from isaacsim.core.experimental.utils.semantics import add_labels

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if OUTPUT_USD.exists():
            OUTPUT_USD.unlink()
        stage = Usd.Stage.CreateNew(str(OUTPUT_USD))
        stage.GetRootLayer().subLayerPaths.append(str(vendor))
        stage.GetRootLayer().Save()
        stage.Reload()
        stage.SetEditTarget(stage.GetRootLayer())
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Root"))

        counts = Counter()
        mesh_count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            label = classify_prim_path(str(prim.GetPath()))
            add_labels(prim, labels=[label], taxonomy="class")
            counts[label] += 1
            mesh_count += 1
        stage.GetRootLayer().Save()

        if mesh_count != phase3["scene_roles"]["primary"]["inventory"]["mesh_count"]:
            raise RuntimeError(f"composed mesh count drifted: {mesh_count}")
        if sum(counts.values()) != mesh_count:
            raise RuntimeError("not every mesh received exactly one class")
        missing = sorted(set(counts) - set(CHANNELS))
        if missing:
            raise RuntimeError(f"overlay contains classes outside {TAXONOMY_ID}: {missing}")

        manifest = {
            "schema_version": "phase4-semantic-overlay-v1",
            "taxonomy_id": TAXONOMY_ID,
            "vendor_usd": str(vendor),
            "vendor_sha256_before": expected_vendor_hash,
            "vendor_sha256_after": sha256(vendor),
            "overlay_usd": str(OUTPUT_USD.relative_to(ROOT)),
            "mesh_count": mesh_count,
            "labeled_mesh_count": sum(counts.values()),
            "stage_metadata": {
                "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                "default_prim": str(stage.GetDefaultPrim().GetPath()),
            },
            "class_mesh_counts": {name: counts.get(name, 0) for name in CHANNELS},
            "mapping_basis": "ordered stable prim-path rules over frozen Isaac Sim 6.0 asset",
        }
        OUTPUT_MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        print(json.dumps(manifest, indent=2))
        print(f"Phase 4 semantic overlay written: {OUTPUT_USD}")
    finally:
        app.close()


if __name__ == "__main__":
    main()
