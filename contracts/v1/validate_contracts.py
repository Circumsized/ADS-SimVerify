#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["PyYAML>=6.0"]
# ///
import argparse
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = Path(__file__).with_name("data_contracts.json")
REQUIRED_STREAM_FIELDS = {
    "version",
    "dtype",
    "shape",
    "fields",
    "unit",
    "frame",
    "nominal_hz",
    "freshness",
    "failure_policy",
}


def topology_edges(path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    edges = set()
    for node in document.get("nodes", []):
        node_id = node["id"]
        for input_id, binding in (node.get("inputs") or {}).items():
            source = binding.get("source") if isinstance(binding, dict) else binding
            edges.add((source, f"{node_id}/{input_id}"))
    return edges


def validate_contract(document, repo_root):
    errors = []
    if document.get("status") != "frozen":
        errors.append("root.status must be 'frozen'")

    streams = document.get("streams", {})
    for stream_id, stream in streams.items():
        missing = sorted(REQUIRED_STREAM_FIELDS - stream.keys())
        if missing:
            errors.append(f"stream {stream_id} missing fields: {', '.join(missing)}")
        if not stream.get("shape"):
            errors.append(f"stream {stream_id} has an empty shape")
        freshness = stream.get("freshness", {})
        if not isinstance(freshness.get("max_age_ms"), (int, float)):
            errors.append(f"stream {stream_id} freshness.max_age_ms must be numeric")

    policy_boundary = document.get("policy_boundary", {})
    for boundary_id, boundary in policy_boundary.items():
        missing = sorted(REQUIRED_STREAM_FIELDS - boundary.keys())
        if missing:
            errors.append(f"policy boundary {boundary_id} missing fields: {', '.join(missing)}")

    topologies = document.get("topologies", {})
    for topology_name, bindings in topologies.items():
        topology_path = repo_root / topology_name
        if not topology_path.is_file():
            errors.append(f"missing topology file: {topology_name}")
            continue
        expected = set()
        for binding in bindings:
            contract_id = binding.get("contract")
            if contract_id not in streams:
                errors.append(
                    f"{topology_name} binding {binding.get('target')} references unknown contract {contract_id}"
                )
            expected.add((binding.get("source"), binding.get("target")))
        actual = topology_edges(topology_path)
        for edge in sorted(actual - expected):
            errors.append(f"{topology_name} undocumented edge: {edge[0]} -> {edge[1]}")
        for edge in sorted(expected - actual):
            errors.append(f"{topology_name} stale contract edge: {edge[0]} -> {edge[1]}")

    deviation_ids = [item.get("id") for item in document.get("known_deviations", [])]
    if len(deviation_ids) != len(set(deviation_ids)):
        errors.append("known deviation ids must be unique")
    for deviation in document.get("known_deviations", []):
        for field in ("id", "severity", "owner_phase", "summary", "evidence"):
            if not deviation.get(field):
                errors.append(f"known deviation missing {field}: {deviation}")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate frozen Phase 1 data contracts")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    document = json.loads(args.contract.read_text(encoding="utf-8"))
    errors = validate_contract(document, args.repo_root.resolve())
    if errors:
        print("Phase 1 contract validation FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    edge_count = sum(len(edges) for edges in document["topologies"].values())
    print(
        "Phase 1 contract validation OK: "
        f"{len(document['streams'])} streams, "
        f"{len(document['policy_boundary'])} policy boundaries, "
        f"{edge_count} topology edges, "
        f"{len(document['known_deviations'])} known deviations"
    )
    if document["deployment_domain"] == "unresolved":
        print("Gate G0 remains open: deployment domain is unresolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
