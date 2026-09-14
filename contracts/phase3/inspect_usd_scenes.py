#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from isaacsim import SimulationApp


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_scene(path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"failed to open USD: {path}")
    types = Counter()
    collision_api_count = 0
    material_binding_api_count = 0
    semantic_label_binding_count = 0
    for prim in stage.Traverse():
        types[prim.GetTypeName() or "<none>"] += 1
        schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
        collision_api_count += any("CollisionAPI" in schema for schema in schemas)
        material_binding_api_count += any("MaterialBindingAPI" in schema for schema in schemas)
        semantic_label_binding_count += sum(
            schema.startswith("SemanticsLabelsAPI:") for schema in schemas
        )
    default_prim = stage.GetDefaultPrim()
    return {
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "prim_count": sum(types.values()),
        "mesh_count": types["Mesh"],
        "collision_api_count": collision_api_count,
        "material_binding_api_count": material_binding_api_count,
        "rect_light_count": types["RectLight"],
        "navmesh_volume_count": types["NavMeshVolume"],
        "semantic_label_binding_count": semantic_label_binding_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect frozen Phase 3 USD scenes with Isaac Sim")
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).with_name("domain_scene_baseline.json")),
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    app = SimulationApp({"headless": True})
    errors = []
    report = {}
    try:
        for role, scene in manifest["scene_roles"].items():
            path = Path(scene["local_path"])
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            actual = inspect_scene(path)
            report[role] = actual
            if sha256(path) != scene["sha256"]:
                errors.append(f"{role}: SHA256 mismatch")
            for key, expected in scene["inventory"].items():
                if actual.get(key) != expected:
                    errors.append(f"{role}: {key} expected={expected!r} actual={actual.get(key)!r}")
        print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", flush=True)
            raise SystemExit(1)
        print("Phase 3 Isaac USD scene inspection OK", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
