#!/usr/bin/env python3
import unittest

import numpy as np

from phase5c_dynamic_upper_bound import dynamic_case, dynamic_mask


class Phase5CDynamicTests(unittest.TestCase):
    def test_schedule_has_four_stop_and_six_go_cases(self):
        cases = [dynamic_case(index)[0] for index in range(10)]
        self.assertEqual(cases.count("center_stop"), 4)
        self.assertEqual(sum(name.endswith("go") or "_go_" in name for name in cases), 6)

    def test_dynamic_mask_is_centered_in_vehicle_bev(self):
        bev = {
            "shape": [192, 192, 14],
            "ego_origin_cell": [95.5, 95.5],
            "meters_per_cell": 20.0 / 192.0,
        }
        mask = dynamic_mask(bev, 0.5, 0.0)
        self.assertTrue(mask.any())
        rows, cols = np.nonzero(mask)
        self.assertLess(float(cols.mean()), 96.0)
        self.assertLess(float(rows.mean()), 95.5)


if __name__ == "__main__":
    unittest.main()
