#!/usr/bin/env python3
import unittest

import numpy as np

from phase5f_dual_nmpc_shadow import bev_obstacle_parameters, brake_metrics


class Phase5FShadowTests(unittest.TestCase):
    def setUp(self):
        self.bev = {
            "shape": [192, 192, 14],
            "meters_per_cell": 20.0 / 192.0,
            "ego_origin_cell": [95.5, 95.5],
        }

    def cell(self, forward, left):
        row = round(self.bev["ego_origin_cell"][0] - forward / self.bev["meters_per_cell"])
        col = round(self.bev["ego_origin_cell"][1] - left / self.bev["meters_per_cell"])
        return row, col

    def test_adapter_selects_nearest_valid_cell_in_each_production_sector(self):
        occupied = np.zeros((192, 192), dtype=bool)
        valid = np.ones_like(occupied)
        for forward, left in ((1.2, 0.4), (0.7, 0.4), (0.5, 0.0), (0.9, -0.4)):
            occupied[self.cell(forward, left)] = True
        parameters, count = bev_obstacle_parameters(occupied, valid, self.bev)
        self.assertEqual(count, 3)
        self.assertAlmostEqual(parameters[0], 0.7, delta=self.bev["meters_per_cell"])
        self.assertAlmostEqual(parameters[4], 0.5, delta=self.bev["meters_per_cell"])
        self.assertAlmostEqual(parameters[8], 0.9, delta=self.bev["meters_per_cell"])

    def test_invalid_occupied_cells_are_not_injected_into_nmpc(self):
        occupied = np.zeros((192, 192), dtype=bool)
        valid = np.ones_like(occupied)
        cell = self.cell(0.5, 0.0)
        occupied[cell] = True
        valid[cell] = False
        parameters, count = bev_obstacle_parameters(occupied, valid, self.bev)
        self.assertEqual(count, 0)
        self.assertTrue(np.all(parameters[[0, 4, 8]] == 1000.0))

    def test_brake_confusion_is_measured_against_oracle(self):
        rows = [
            {"dynamic_mode": "center_stop", "oracle_acceleration_mps2": -0.8, "candidate_brake": True},
            {"dynamic_mode": "center_stop", "oracle_acceleration_mps2": -0.8, "candidate_brake": False},
            {"dynamic_mode": "absent_go", "oracle_acceleration_mps2": 0.2, "candidate_brake": False},
            {"dynamic_mode": "absent_go", "oracle_acceleration_mps2": 0.2, "candidate_brake": True},
            {"dynamic_mode": "center_stop", "oracle_acceleration_mps2": -0.2, "candidate_brake": True},
        ]
        metrics = brake_metrics(rows)
        self.assertEqual(metrics["true_brake"], 1)
        self.assertEqual(metrics["false_brake"], 1)
        self.assertEqual(metrics["oracle_brake_recall"], 0.5)
        self.assertEqual(metrics["oracle_release_specificity"], 0.5)
        self.assertEqual(metrics["moderate_oracle_deceleration_excluded"], 1)


if __name__ == "__main__":
    unittest.main()
