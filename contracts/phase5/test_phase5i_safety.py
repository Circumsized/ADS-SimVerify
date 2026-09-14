#!/usr/bin/env python3
import math
import unittest

from phase5i_safety import ControlProposal, RuntimeHealth, SafetyState, SafetySupervisor


def healthy(frame_id, timestamp_ms):
    return RuntimeHealth(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        sensor_valid=True,
        perception_valid=True,
        solver_valid=True,
        articulation_ready=True,
    )


def proposal(frame_id, timestamp_ms, linear=0.3, angular=0.2):
    return ControlProposal(frame_id, timestamp_ms, linear, angular)


class Phase5ISafetyTests(unittest.TestCase):
    def test_startup_requires_ordered_healthy_warmup_before_motion(self):
        supervisor = SafetySupervisor(warmup_frames=3, watchdog_ms=150)

        first = supervisor.step(0, healthy(1, 0), proposal(1, 0))
        second = supervisor.step(50, healthy(2, 50), proposal(2, 50))
        ready = supervisor.step(100, healthy(3, 100), proposal(3, 100))
        active = supervisor.step(150, healthy(4, 150), proposal(4, 150))

        self.assertEqual(first.state, SafetyState.WARMUP)
        self.assertEqual(second.state, SafetyState.WARMUP)
        self.assertEqual(ready.state, SafetyState.READY)
        self.assertEqual((first.linear, second.linear, ready.linear), (0.0, 0.0, 0.0))
        self.assertEqual(active.state, SafetyState.ACTIVE)
        self.assertEqual((active.linear, active.angular), (0.3, 0.2))

    def test_emergency_stop_latches_until_explicit_reset_and_rewarm(self):
        supervisor = SafetySupervisor(warmup_frames=2, watchdog_ms=150)
        supervisor.step(0, healthy(1, 0), proposal(1, 0))
        supervisor.step(50, healthy(2, 50), proposal(2, 50))
        supervisor.step(100, healthy(3, 100), proposal(3, 100))

        stopped = supervisor.step(150, healthy(4, 150), proposal(4, 150), emergency_stop=True)
        still_stopped = supervisor.step(200, healthy(5, 200), proposal(5, 200))
        reset = supervisor.step(250, healthy(6, 250), proposal(6, 250), reset=True)
        ready = supervisor.step(300, healthy(7, 300), proposal(7, 300))
        active = supervisor.step(350, healthy(8, 350), proposal(8, 350))

        self.assertEqual(stopped.state, SafetyState.EMERGENCY_STOP)
        self.assertTrue(stopped.latched)
        self.assertEqual(still_stopped.state, SafetyState.EMERGENCY_STOP)
        self.assertEqual(reset.state, SafetyState.WARMUP)
        self.assertEqual(ready.state, SafetyState.READY)
        self.assertEqual(active.state, SafetyState.ACTIVE)
        self.assertEqual((active.linear, active.angular), (0.3, 0.2))

    def test_watchdog_latches_fault_for_stale_command(self):
        supervisor = SafetySupervisor(warmup_frames=1, watchdog_ms=120)
        supervisor.step(0, healthy(1, 0), proposal(1, 0))
        supervisor.step(50, healthy(2, 50), proposal(2, 50))

        fault = supervisor.step(200, healthy(3, 200), proposal(3, 50))

        self.assertEqual(fault.state, SafetyState.FAULT)
        self.assertEqual(fault.reason, "stale_command")
        self.assertEqual((fault.linear, fault.angular), (0.0, 0.0))
        self.assertTrue(fault.latched)

    def test_invalid_health_frame_order_and_nonfinite_command_fail_closed(self):
        supervisor = SafetySupervisor(warmup_frames=1, watchdog_ms=120)
        supervisor.step(0, healthy(1, 0), proposal(1, 0))
        supervisor.step(50, healthy(2, 50), proposal(2, 50))

        mismatch = supervisor.step(100, healthy(3, 100), proposal(4, 100))
        self.assertEqual(mismatch.state, SafetyState.FAULT)
        self.assertEqual(mismatch.reason, "frame_mismatch")

        supervisor = SafetySupervisor(warmup_frames=1, watchdog_ms=120)
        supervisor.step(0, healthy(1, 0), proposal(1, 0))
        supervisor.step(50, healthy(2, 50), proposal(2, 50))
        nonfinite = supervisor.step(100, healthy(3, 100), proposal(3, 100, math.nan, 0.0))
        self.assertEqual(nonfinite.state, SafetyState.FAULT)
        self.assertEqual(nonfinite.reason, "invalid_command")

    def test_shutdown_is_terminal_and_outputs_zero(self):
        supervisor = SafetySupervisor(warmup_frames=1, watchdog_ms=120)
        supervisor.step(0, healthy(1, 0), proposal(1, 0))
        supervisor.step(50, healthy(2, 50), proposal(2, 50))

        stopped = supervisor.step(100, shutdown=True)
        after_stop = supervisor.step(150, healthy(3, 150), proposal(3, 150))

        self.assertEqual(stopped.state, SafetyState.STOPPED)
        self.assertEqual(after_stop.state, SafetyState.STOPPED)
        self.assertEqual((after_stop.linear, after_stop.angular), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
