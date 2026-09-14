#!/usr/bin/env python3
"""Frozen warehouse_nav14_v1 taxonomy and deterministic USD path mapping."""

TAXONOMY_ID = "warehouse_nav14_v1"
CHANNELS = (
    "traversable_floor",
    "floor_marking",
    "curb_or_step",
    "wall_or_door",
    "fence_or_guardrail",
    "shelf_or_rack",
    "pole_or_column",
    "traffic_control_or_sign",
    "pallet_or_load",
    "box_or_small_obstacle",
    "person",
    "robot_or_cart",
    "forklift_or_heavy_vehicle",
    "unknown_or_unlabeled",
)
CHANNEL_BY_NAME = {name: index for index, name in enumerate(CHANNELS)}
FREE_CHANNELS = frozenset((0, 1))

# Ordered from specific to general. The vendor asset is immutable, so matching
# uses stable composed prim paths and is audited against the frozen scene hash.
PATH_RULES = (
    ("floor_marking", ("floordecal", "stripefull", "floorstripe", "marking")),
    ("traversable_floor", ("groundplane", "sm_floor")),
    ("person", ("person", "human", "worker", "pedestrian")),
    ("forklift_or_heavy_vehicle", ("forklift", "heavyvehicle", "truck")),
    ("robot_or_cart", ("pushcart", "cart", "robot", "jetbot")),
    (
        "box_or_small_obstacle",
        ("smallklt", "cardbox", "crate", "fireextinguisher", "fusebox", "box"),
    ),
    ("pallet_or_load", ("pallet", "palette", "load")),
    (
        "traffic_control_or_sign",
        ("aislesign", "barcode", "sm_sign", "signa", "signb", "signc"),
    ),
    ("fence_or_guardrail", ("rackshield", "guardrail", "fence", "wallwire")),
    ("shelf_or_rack", ("rackshelf", "rackframe", "shelf", "roller", "palletbin")),
    ("pole_or_column", ("pillar", "column", "pole")),
    ("wall_or_door", ("wall", "door")),
)


def classify_prim_path(path):
    lowered = path.lower()
    for label, tokens in PATH_RULES:
        if any(token in lowered for token in tokens):
            return label
    return "unknown_or_unlabeled"


def channel_id(label):
    return CHANNEL_BY_NAME.get(label, CHANNEL_BY_NAME["unknown_or_unlabeled"])
