#!/usr/bin/env python3
from dataclasses import dataclass


RUN_MODES = {
    "endurance",
    "controller_sigkill",
    "supervisor_sigkill",
    "sensor_freeze",
    "restart_recovery",
}


@dataclass(frozen=True)
class FaultAction:
    kill_target: str | None = None
    freeze_sensor: bool | None = None
    reset: bool = False

    @property
    def has_action(self):
        return self.kill_target is not None or self.freeze_sensor is not None or self.reset


def fault_action(run_mode, frame_id):
    if run_mode not in RUN_MODES:
        raise ValueError(f"unsupported run mode: {run_mode}")
    frame_id = int(frame_id)
    if run_mode == "controller_sigkill" and frame_id == 80:
        return FaultAction(kill_target="controller")
    if run_mode == "supervisor_sigkill" and frame_id == 80:
        return FaultAction(kill_target="supervisor")
    if run_mode == "sensor_freeze":
        if frame_id == 80:
            return FaultAction(freeze_sensor=True)
        if frame_id == 88:
            return FaultAction(freeze_sensor=False)
        if frame_id == 92:
            return FaultAction(reset=True)
    return FaultAction()


class SensorSequenceGuard:
    def __init__(self, expected_generation):
        self.expected_generation = int(expected_generation)
        self.frame_id = None
        self.timestamp_ms = None

    def observe(self, generation, frame_id, timestamp_ms):
        if int(generation) != self.expected_generation:
            return False, "wrong_generation"
        frame_id = int(frame_id)
        timestamp_ms = float(timestamp_ms)
        if self.frame_id is not None and frame_id <= self.frame_id:
            return False, "sensor_replay"
        if self.timestamp_ms is not None and timestamp_ms <= self.timestamp_ms:
            return False, "stale_sensor_timestamp"
        self.frame_id = frame_id
        self.timestamp_ms = timestamp_ms
        return True, "fresh"


class CommandGenerationGuard:
    def __init__(self, expected_generation):
        self.expected_generation = int(expected_generation)
        self.sequence_id = None

    def accept(self, generation, sequence_id):
        generation = int(generation)
        sequence_id = int(sequence_id)
        if generation != self.expected_generation:
            return False
        if self.sequence_id is not None and sequence_id <= self.sequence_id:
            return False
        self.sequence_id = sequence_id
        return True
