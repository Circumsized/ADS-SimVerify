#!/usr/bin/env python3
import unittest

from warehouse_semantics import CHANNELS, FREE_CHANNELS, channel_id, classify_prim_path


class WarehouseSemanticsTest(unittest.TestCase):
    def test_frozen_shape_and_free_channels(self):
        self.assertEqual(len(CHANNELS), 14)
        self.assertEqual(FREE_CHANNELS, {0, 1})

    def test_specific_rules_precede_container_rules(self):
        self.assertEqual(
            classify_prim_path("/Root/Shelf_0/CardBoxSet1/mesh"),
            "box_or_small_obstacle",
        )
        self.assertEqual(
            classify_prim_path("/Root/Palette02BinSet1/SmallKLT/mesh"),
            "box_or_small_obstacle",
        )
        self.assertEqual(
            classify_prim_path("/Root/Shelf_0/SM_RackShelf_01/mesh"),
            "shelf_or_rack",
        )

    def test_floor_and_unknown_are_explicit(self):
        self.assertEqual(classify_prim_path("/Root/SM_floor27/mesh"), "traversable_floor")
        self.assertEqual(classify_prim_path("/Root/SM_CeilingA/mesh"), "unknown_or_unlabeled")
        self.assertEqual(channel_id("not-a-class"), 13)


if __name__ == "__main__":
    unittest.main()
