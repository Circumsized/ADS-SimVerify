#!/usr/bin/env python3
import math
from copy import deepcopy
from dataclasses import dataclass


def dual_resolution_geometry(source, control_size=(320, 240)):
    sensor = deepcopy(source)
    control = deepcopy(source)
    source_width, source_height = source["image_size"]
    width, height = control_size
    scale_x, scale_y = width / source_width, height / source_height
    control["image_size"] = [width, height]
    for name in ("fx", "cx"):
        control["intrinsics"][name] *= scale_x
    for name in ("fy", "cy"):
        control["intrinsics"][name] *= scale_y
    return sensor, control


@dataclass(frozen=True)
class ActuatorCommand:
    linear: float
    angular: float
    reason: str


class ActuatorWatchdog:
    def __init__(self, timeout_ms=150, max_linear=0.8, max_angular=0.6):
        self.timeout_ms = float(timeout_ms)
        self.max_linear = float(max_linear)
        self.max_angular = float(max_angular)
        self.frame_id = None
        self.received_ms = None
        self.linear = 0.0
        self.angular = 0.0

    def update(self, frame_id, received_ms, linear, angular):
        values = (received_ms, linear, angular)
        if not all(math.isfinite(float(value)) for value in values):
            return False
        if self.frame_id is not None and int(frame_id) <= self.frame_id:
            return False
        if not 0.0 <= float(linear) <= self.max_linear or abs(float(angular)) > self.max_angular:
            return False
        self.frame_id = int(frame_id)
        self.received_ms = float(received_ms)
        self.linear = float(linear)
        self.angular = float(angular)
        return True

    def command(self, now_ms):
        if self.received_ms is None or float(now_ms) - self.received_ms > self.timeout_ms:
            return ActuatorCommand(0.0, 0.0, "actuator_watchdog_timeout")
        return ActuatorCommand(self.linear, self.angular, "fresh")


def proposal_timestamp_ms(run_mode, frame_id, now_ms, watchdog_ms):
    if run_mode == "watchdog_reset" and int(frame_id) == 60:
        return float(now_ms) - float(watchdog_ms) - 1.0
    return float(now_ms)


def localization_due(frame_id, control_rate_hz=20, localization_rate_hz=2):
    if control_rate_hz <= 0 or localization_rate_hz <= 0:
        raise ValueError("sensor rates must be positive")
    period = int(control_rate_hz) // int(localization_rate_hz)
    if period < 1 or period * int(localization_rate_hz) != int(control_rate_hz):
        raise ValueError("control rate must be an integer multiple of localization rate")
    return int(frame_id) % period == 0
