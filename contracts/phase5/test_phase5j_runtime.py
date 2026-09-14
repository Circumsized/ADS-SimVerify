#!/usr/bin/env python3
import subprocess
import struct
import unittest

from phase5j_runtime import FrameTripletBuffer, episode_for_frame, sigkill_and_confirm, transport_safe_clip


class Phase5JRuntimeTests(unittest.TestCase):
    def test_triplet_buffer_emits_only_exact_rgb_depth_state_frame(self):
        pairer = FrameTripletBuffer(max_pending=2)

        self.assertIsNone(pairer.add("rgb", 7, "rgb7"))
        self.assertIsNone(pairer.add("depth", 8, "depth8"))
        self.assertIsNone(pairer.add("state", 7, "state7"))
        packet = pairer.add("depth", 7, "depth7")

        self.assertEqual(packet, {"rgb": "rgb7", "depth": "depth7", "state": "state7"})
        self.assertNotIn(7, pairer.pending)

    def test_triplet_buffer_bounds_unmatched_sensor_memory(self):
        pairer = FrameTripletBuffer(max_pending=2)
        pairer.add("rgb", 1, "one")
        pairer.add("rgb", 2, "two")
        pairer.add("rgb", 3, "three")

        self.assertEqual(list(pairer.pending), [2, 3])

    def test_episode_schedule_repeats_all_four_scenarios_in_order(self):
        scenarios = ("straight", "diagonal", "pallet", "cart")

        self.assertEqual(episode_for_frame(0, 10, scenarios), (0, "straight", 0))
        self.assertEqual(episode_for_frame(39, 10, scenarios), (3, "cart", 9))
        self.assertEqual(episode_for_frame(40, 10, scenarios), (4, "straight", 0))

    def test_sigkill_confirmation_observes_real_process_death(self):
        child = subprocess.Popen(["sleep", "30"])
        try:
            self.assertTrue(sigkill_and_confirm(child.pid, timeout_s=1.0))
            self.assertEqual(child.wait(timeout=1.0), -9)
        finally:
            if child.poll() is None:
                child.kill()

    def test_transport_clip_remains_inside_limit_after_float32_serialization(self):
        clipped = transport_safe_clip(-0.6, 0.6)
        transported = struct.unpack("f", struct.pack("f", clipped))[0]

        self.assertLessEqual(abs(transported), 0.6)
        self.assertLess(transported, 0.0)


if __name__ == "__main__":
    unittest.main()
