#!/usr/bin/env python3
import unittest

from phase5h_articulation_takeover import (
    WHEEL_BASE_M,
    WHEEL_RADIUS_M,
    differential_wheel_targets,
    scaled_sensor_geometry,
)


class Phase5HArticulationTests(unittest.TestCase):
    def test_scaled_sensor_preserves_normalized_intrinsics(self):
        source = {
            "image_size": [640, 480],
            "intrinsics": {"fx": 204.0, "fy": 154.0, "cx": 319.5, "cy": 239.5},
        }
        scaled = scaled_sensor_geometry(source)
        self.assertEqual(scaled["image_size"], [320, 240])
        self.assertAlmostEqual(scaled["intrinsics"]["fx"] / 320, source["intrinsics"]["fx"] / 640)
        self.assertAlmostEqual(scaled["intrinsics"]["fy"] / 240, source["intrinsics"]["fy"] / 480)
        self.assertEqual(source["image_size"], [640, 480])

    def test_differential_targets_encode_forward_and_yaw(self):
        left, right = differential_wheel_targets(0.3, 0.0)
        self.assertAlmostEqual(left, 0.3 / WHEEL_RADIUS_M)
        self.assertAlmostEqual(right, left)
        left, right = differential_wheel_targets(0.0, 0.6)
        self.assertAlmostEqual(left, -0.6 * WHEEL_BASE_M / (2.0 * WHEEL_RADIUS_M))
        self.assertAlmostEqual(right, -left)


if __name__ == "__main__":
    unittest.main()
