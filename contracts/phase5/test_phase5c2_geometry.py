#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_perception_oracle_map import clip_polygon_height_band
from depth_bev_geometry import depth_lift_semantic_bev
from phase5c2_geometry_upper_bound import dynamic_footprint_mask


class Phase5C2GeometryTests(unittest.TestCase):
    def test_height_band_clips_vertical_face(self):
        face = [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0), (1.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
        clipped = clip_polygon_height_band(face, 0.02, 0.35)
        self.assertGreaterEqual(len(clipped), 4)
        self.assertTrue(all(0.02 <= point[2] <= 0.35 for point in clipped))

    def test_overhead_obstacle_point_is_not_projected_to_ground(self):
        sensor = {
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 1.0},
            "body_extrinsics": {
                "forward_m": 0.0,
                "left_m": 0.0,
                "height_m": 0.2,
                "yaw_rad": 0.0,
                "pitch_rad": 0.0,
                "roll_rad": 0.0,
            },
        }
        bev = {
            "shape": [20, 20, 14],
            "meters_per_cell": 0.1,
            "ego_origin_cell": [10.0, 10.0],
        }
        semantic = np.full((2, 1), 5, dtype=np.uint8)
        depth = np.ones((2, 1), dtype=np.float32)
        _, observed = depth_lift_semantic_bev(semantic, depth, sensor, bev)
        self.assertEqual(int(observed.sum()), 1)

    def test_dynamic_footprint_includes_intersecting_boundary_cell(self):
        bev = {
            "shape": [192, 192, 14],
            "meters_per_cell": 20.0 / 192.0,
            "ego_origin_cell": [95.5, 95.5],
        }
        footprint = dynamic_footprint_mask(bev, 0.5, 0.0)
        row = int(round(bev["ego_origin_cell"][0] - 0.275 / bev["meters_per_cell"]))
        col = int(round(bev["ego_origin_cell"][1]))
        self.assertTrue(footprint[row, col])


if __name__ == "__main__":
    unittest.main()
