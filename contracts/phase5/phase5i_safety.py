#!/usr/bin/env python3
import math
from dataclasses import dataclass
from enum import Enum


class SafetyState(str, Enum):
    BOOT = "boot"
    WARMUP = "warmup"
    READY = "ready"
    ACTIVE = "active"
    EMERGENCY_STOP = "emergency_stop"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True)
class RuntimeHealth:
    frame_id: int
    timestamp_ms: float
    sensor_valid: bool
    perception_valid: bool
    solver_valid: bool
    articulation_ready: bool


@dataclass(frozen=True)
class ControlProposal:
    frame_id: int
    timestamp_ms: float
    linear: float
    angular: float


@dataclass(frozen=True)
class SafetyDecision:
    state: SafetyState
    linear: float
    angular: float
    reason: str
    latched: bool


class SafetySupervisor:
    def __init__(self, warmup_frames=3, watchdog_ms=150, max_linear=0.8, max_angular=0.6):
        if warmup_frames < 1 or watchdog_ms <= 0:
            raise ValueError("warmup_frames and watchdog_ms must be positive")
        self.warmup_frames = int(warmup_frames)
        self.watchdog_ms = float(watchdog_ms)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.state = SafetyState.BOOT
        self.reason = "boot"
        self.healthy_frames = 0
        self.last_frame_id = None

    def _zero(self):
        return SafetyDecision(
            self.state,
            0.0,
            0.0,
            self.reason,
            self.state in (SafetyState.EMERGENCY_STOP, SafetyState.FAULT),
        )

    def _fault(self, reason):
        self.state = SafetyState.FAULT
        self.reason = reason
        return self._zero()

    def _validate(self, now_ms, health, command):
        if health is None or command is None:
            return "missing_runtime_input"
        if not all((health.sensor_valid, health.perception_valid, health.solver_valid, health.articulation_ready)):
            return "runtime_unhealthy"
        if health.frame_id != command.frame_id:
            return "frame_mismatch"
        if self.last_frame_id is not None and health.frame_id != self.last_frame_id + 1:
            return "frame_sequence_gap"
        if not math.isfinite(health.timestamp_ms) or now_ms < health.timestamp_ms:
            return "invalid_health_timestamp"
        if now_ms - health.timestamp_ms > self.watchdog_ms:
            return "stale_health"
        if not math.isfinite(command.timestamp_ms) or now_ms < command.timestamp_ms:
            return "invalid_command_timestamp"
        if now_ms - command.timestamp_ms > self.watchdog_ms:
            return "stale_command"
        if health.timestamp_ms != command.timestamp_ms:
            return "timestamp_mismatch"
        values = (command.linear, command.angular)
        if not all(math.isfinite(value) for value in values):
            return "invalid_command"
        if not 0.0 <= command.linear <= self.max_linear or abs(command.angular) > self.max_angular:
            return "invalid_command"
        return None

    def step(self, now_ms, health=None, command=None, emergency_stop=False, reset=False, shutdown=False):
        if self.state == SafetyState.STOPPED:
            return self._zero()
        if shutdown:
            self.state = SafetyState.STOPPED
            self.reason = "shutdown"
            return self._zero()
        if emergency_stop:
            self.state = SafetyState.EMERGENCY_STOP
            self.reason = "emergency_stop"
            return self._zero()
        if self.state in (SafetyState.EMERGENCY_STOP, SafetyState.FAULT):
            if not reset:
                return self._zero()
            self.state = SafetyState.BOOT
            self.reason = "reset"
            self.healthy_frames = 0
            self.last_frame_id = None

        invalid = self._validate(float(now_ms), health, command)
        if invalid:
            return self._fault(invalid)
        self.last_frame_id = health.frame_id

        if self.state in (SafetyState.BOOT, SafetyState.WARMUP):
            self.healthy_frames += 1
            if self.healthy_frames < self.warmup_frames:
                self.state = SafetyState.WARMUP
                self.reason = "warming_up"
            else:
                self.state = SafetyState.READY
                self.reason = "startup_ready"
            return self._zero()
        if self.state == SafetyState.READY:
            self.state = SafetyState.ACTIVE
        self.reason = "allow"
        return SafetyDecision(self.state, command.linear, command.angular, self.reason, False)
