#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5d_runtime_node import FramePairer


class Phase5DRuntimeTests(unittest.TestCase):
    def test_pairer_requires_exact_source_frame_id(self):
        pairer = FramePairer(4)
        pair, evicted = pairer.put("jpeg_image", 7, b"jpeg")
        self.assertIsNone(pair)
        self.assertEqual(evicted, [])
        pair, evicted = pairer.put("metric_depth", 8, b"wrong depth")
        self.assertIsNone(pair)
        pair, evicted = pairer.put("metric_depth", 7, b"depth")
        self.assertEqual(pair, (7, b"jpeg", b"depth"))
        self.assertEqual(evicted, [])

    def test_pairer_evicts_unmatched_frames_at_bound(self):
        pairer = FramePairer(2)
        pairer.put("jpeg_image", 1, b"a")
        pairer.put("jpeg_image", 2, b"b")
        pair, evicted = pairer.put("jpeg_image", 3, b"c")
        self.assertIsNone(pair)
        self.assertEqual(evicted, [1])


if __name__ == "__main__":
    unittest.main()
