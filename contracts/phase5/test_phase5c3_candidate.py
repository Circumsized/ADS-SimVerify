#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5c3_candidate_shadow import candidate_depth_lift


class Phase5C3CandidateTests(unittest.TestCase):
    def test_depth_support_is_independent_of_predicted_class(self):
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
        free_classes = np.zeros((2, 1), dtype=np.uint8)
        occupied_classes = np.full((2, 1), 12, dtype=np.uint8)
        depth = np.ones((2, 1), dtype=np.float32)
        free_lifted, free_observed = candidate_depth_lift(
            free_classes, depth, sensor, bev
        )
        occupied_lifted, occupied_observed = candidate_depth_lift(
            occupied_classes, depth, sensor, bev
        )
        np.testing.assert_array_equal(free_observed, occupied_observed)
        self.assertEqual(int(free_observed.sum()), 1)
        self.assertTrue(bool(free_lifted[0][free_observed].all()))
        self.assertTrue(bool(occupied_lifted[12][occupied_observed].all()))

    def test_unknown_class_remains_conservative_occupied(self):
        sensor = {
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
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
        classes = np.asarray([[13]], dtype=np.uint8)
        depth = np.ones((1, 1), dtype=np.float32)
        lifted, observed = candidate_depth_lift(classes, depth, sensor, bev)
        self.assertEqual(int(observed.sum()), 1)
        self.assertTrue(bool(lifted[13][observed].all()))


if __name__ == "__main__":
    unittest.main()
