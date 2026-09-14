#!/usr/bin/env python3
import unittest

from phase5i_safety import ControlProposal, RuntimeHealth, SafetyState
from phase5j_safety_relay import SafetyRelay


def health(frame_id, timestamp_ms):
    return RuntimeHealth(frame_id, timestamp_ms, True, True, True, True)


def proposal(frame_id, timestamp_ms):
    return ControlProposal(frame_id, timestamp_ms, 0.3, 0.1)


class Phase5JSafetyRelayTests(unittest.TestCase):
    def test_back_to_back_proposals_are_processed_immediately_without_frame_skip(self):
        relay = SafetyRelay(warmup_frames=1, watchdog_ms=150)

        ready = relay.update(0.0, health(0, 0.0), proposal(0, 0.0), sensor_started_ns=10)
        active = relay.update(1.0, health(1, 1.0), proposal(1, 1.0), sensor_started_ns=20)

        self.assertEqual(ready.decision.state, SafetyState.READY)
        self.assertEqual(active.decision.state, SafetyState.ACTIVE)
        self.assertEqual(active.source_frame_id, 1)
        self.assertTrue(active.proposal_fresh)

    def test_timer_repeats_fresh_decision_then_fails_closed_at_watchdog_boundary(self):
        relay = SafetyRelay(warmup_frames=1, watchdog_ms=150)
        relay.update(0.0, health(0, 0.0), proposal(0, 0.0), sensor_started_ns=10)
        relay.update(1.0, health(1, 1.0), proposal(1, 1.0), sensor_started_ns=20)

        fresh = relay.tick(150.9)
        stale = relay.tick(151.0)

        self.assertEqual(fresh.decision.state, SafetyState.ACTIVE)
        self.assertFalse(fresh.proposal_fresh)
        self.assertEqual(stale.decision.state, SafetyState.FAULT)
        self.assertEqual((stale.decision.linear, stale.decision.angular), (0.0, 0.0))

    def test_fault_reset_requires_a_new_proposal_and_rewarm(self):
        relay = SafetyRelay(warmup_frames=1, watchdog_ms=100)
        relay.update(0.0, health(0, 0.0), proposal(0, 0.0), sensor_started_ns=10)
        relay.update(1.0, health(1, 1.0), proposal(1, 1.0), sensor_started_ns=20)
        relay.tick(101.0)

        relay.request_reset()
        still_fault = relay.tick(102.0)
        ready = relay.update(103.0, health(5, 103.0), proposal(5, 103.0), sensor_started_ns=30)
        active = relay.update(104.0, health(6, 104.0), proposal(6, 104.0), sensor_started_ns=40)

        self.assertEqual(still_fault.decision.state, SafetyState.FAULT)
        self.assertEqual(ready.decision.state, SafetyState.READY)
        self.assertTrue(ready.reset_applied)
        self.assertEqual(active.decision.state, SafetyState.ACTIVE)


if __name__ == "__main__":
    unittest.main()
