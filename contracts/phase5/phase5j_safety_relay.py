#!/usr/bin/env python3
from dataclasses import dataclass

from phase5i_safety import SafetyDecision, SafetyState, SafetySupervisor


@dataclass(frozen=True)
class RelayOutput:
    decision: SafetyDecision
    source_frame_id: int
    sensor_started_ns: int
    reset_applied: bool
    proposal_fresh: bool


class SafetyRelay:
    def __init__(self, warmup_frames=5, watchdog_ms=150):
        self.supervisor = SafetySupervisor(warmup_frames=warmup_frames, watchdog_ms=watchdog_ms)
        self.output = RelayOutput(
            SafetyDecision(SafetyState.BOOT, 0.0, 0.0, "boot", False),
            -1,
            -1,
            False,
            False,
        )
        self.last_update_ms = None
        self.reset_pending = False
        self.emergency_stop = False

    def request_reset(self):
        self.reset_pending = True

    def set_emergency_stop(self, enabled):
        self.emergency_stop = bool(enabled)

    def update(self, now_ms, health, proposal, sensor_started_ns):
        reset_applied = self.reset_pending and self.supervisor.state in (
            SafetyState.FAULT,
            SafetyState.EMERGENCY_STOP,
        )
        decision = self.supervisor.step(
            now_ms,
            health,
            proposal,
            emergency_stop=self.emergency_stop,
            reset=self.reset_pending,
        )
        self.reset_pending = False
        self.last_update_ms = float(now_ms)
        self.output = RelayOutput(
            decision,
            int(proposal.frame_id),
            int(sensor_started_ns),
            reset_applied,
            True,
        )
        return self.output

    def tick(self, now_ms):
        if self.last_update_ms is None:
            return self.output
        if (
            self.supervisor.state not in (SafetyState.FAULT, SafetyState.EMERGENCY_STOP)
            and float(now_ms) - self.last_update_ms >= self.supervisor.watchdog_ms
        ):
            decision = self.supervisor.step(float(now_ms), emergency_stop=self.emergency_stop)
            self.output = RelayOutput(
                decision,
                self.output.source_frame_id,
                self.output.sensor_started_ns,
                False,
                False,
            )
        if self.output.proposal_fresh:
            self.output = RelayOutput(
                self.output.decision,
                self.output.source_frame_id,
                self.output.sensor_started_ns,
                self.output.reset_applied,
                False,
            )
        return self.output
