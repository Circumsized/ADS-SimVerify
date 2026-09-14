#!/usr/bin/env python3
import math
import unittest

import numpy as np

from phase5g_controlled_takeover import (
    DT,
    dynamic_pose,
    integrate_state,
    obb_aabb_collision,
    supervise_swept_step,
)


class FakeGrid:
    half_length = 0.20
    half_width = 0.15
    margin = 0.05

    def footprint_collision(self, x, y, yaw):
        return x >= 1.0


class Phase5GTakeoverTests(unittest.TestCase):
    def test_integrator_uses_20hz_acceleration_state(self):
        state = integrate_state(np.asarray([0.0, 0.0, 0.0, 0.0]), 1.0, 0.0)
        self.assertAlmostEqual(state[3], DT)
        self.assertAlmostEqual(state[0], 0.5 * DT * DT)

    def test_dynamic_cart_follows_continuous_world_trajectory(self):
        self.assertEqual(dynamic_pose("straight_aisle", 3.0), (1000.0, 1000.0))
        self.assertEqual(dynamic_pose("crossing_cart", 0.0), (3.8, -0.5))
        self.assertAlmostEqual(dynamic_pose("crossing_cart", 5.0)[1], -2.0)
        self.assertAlmostEqual(dynamic_pose("crossing_cart", 20.0)[1], -2.6)

    def test_oriented_robot_collision_detects_overlap_and_separation(self):
        self.assertTrue(obb_aabb_collision(0.0, 0.0, math.pi / 4, 0.3, 0.2, 0.2, 0.0, 0.2))
        self.assertFalse(obb_aabb_collision(0.0, 0.0, math.pi / 4, 0.3, 0.2, 1.0, 0.0, 0.2))

    def test_supervisor_aborts_swept_static_collision_without_override(self):
        current = np.asarray([0.9, 0.0, 0.0, 0.2])
        proposed = np.asarray([1.1, 0.0, 0.0, 0.2])
        decision = supervise_swept_step(FakeGrid(), current, proposed, "straight_aisle", 0.0)
        self.assertEqual(decision, "abort_static_or_bounds")


if __name__ == "__main__":
    unittest.main()
