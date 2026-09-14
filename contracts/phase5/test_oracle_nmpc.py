#!/usr/bin/env python3
import math
import unittest

import numpy as np

from oracle_nmpc_closed_loop import Polyline, astar, wrap_angle


class FakeGrid:
    resolution = 1.0
    min_x = 0.0
    max_y = 6.0

    def __init__(self):
        self.inflated = np.zeros((7, 7), dtype=np.uint8)
        self.inflated[3, 1:6] = 1

    def world_to_grid(self, x, y):
        return int(round(self.max_y - y)), int(round(x))

    def grid_to_world(self, row, col):
        return float(col), float(self.max_y - row)

    def inside(self, row, col):
        return 0 <= row < 7 and 0 <= col < 7


class OraclePlannerTests(unittest.TestCase):
    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3.0 * math.pi), -math.pi)
        self.assertAlmostEqual(wrap_angle(-0.25), -0.25)

    def test_astar_routes_around_occupied_barrier(self):
        grid = FakeGrid()
        path = astar(grid, (0.0, 3.0), (6.0, 3.0))
        self.assertGreater(len(path), 7)
        self.assertTrue(all(not grid.inflated[grid.world_to_grid(*point)] for point in path))

    def test_polyline_projection_and_terminal_heading(self):
        path = Polyline([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)])
        self.assertAlmostEqual(path.nearest_s(1.0, 0.2), 1.0)
        self.assertAlmostEqual(path.distance_to(1.0, 0.2), 0.2)
        x, y, yaw = path.pose(path.length, math.pi / 2.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(yaw, math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
