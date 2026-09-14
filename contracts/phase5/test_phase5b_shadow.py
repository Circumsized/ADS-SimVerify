#!/usr/bin/env python3
import unittest

import numpy as np

from phase5b_shadow_replay import decode_pidnet, occupancy_metrics


class Phase5BShadowTests(unittest.TestCase):
    def test_metrics_identical(self):
        occupied = np.array([[False, True], [True, False]])
        metrics = occupancy_metrics(occupied, occupied, np.ones((2, 2), dtype=bool))
        self.assertEqual(metrics["occupied_iou"], 1.0)
        self.assertEqual(metrics["free_iou"], 1.0)
        self.assertEqual(metrics["false_free_rate"], 0.0)
        self.assertEqual(metrics["false_occupied_rate"], 0.0)

    def test_metrics_separate_dangerous_and_nuisance_errors(self):
        oracle = np.array([[True, False]])
        candidate = np.array([[False, True]])
        metrics = occupancy_metrics(oracle, candidate, np.ones((1, 2), dtype=bool))
        self.assertEqual(metrics["false_free_rate"], 1.0)
        self.assertEqual(metrics["false_occupied_rate"], 1.0)

    def test_pidnet_confidence_threshold_matches_deployment(self):
        logits = np.zeros((1, 19, 1, 2), dtype=np.float32)
        logits[0, 3, 0, 0] = 10.0
        decoded = decode_pidnet(logits)
        self.assertEqual(int(decoded[0, 0]), 3)
        self.assertEqual(int(decoded[0, 1]), 255)


if __name__ == "__main__":
    unittest.main()
