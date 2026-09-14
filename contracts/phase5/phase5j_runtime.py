#!/usr/bin/env python3
import os
import signal
import struct
import time
from collections import OrderedDict
from pathlib import Path


class FrameTripletBuffer:
    KINDS = {"rgb", "depth", "state"}

    def __init__(self, max_pending=4):
        self.max_pending = int(max_pending)
        self.pending = OrderedDict()

    def add(self, kind, frame_id, value):
        if kind not in self.KINDS:
            raise ValueError(f"unsupported packet kind: {kind}")
        frame_id = int(frame_id)
        packet = self.pending.setdefault(frame_id, {})
        packet[kind] = value
        self.pending.move_to_end(frame_id)
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)
        if set(packet) != self.KINDS:
            return None
        self.pending.pop(frame_id)
        return packet


def episode_for_frame(frame_id, frames_per_episode, scenario_names):
    frame_id = int(frame_id)
    frames_per_episode = int(frames_per_episode)
    if frame_id < 0 or frames_per_episode <= 0 or not scenario_names:
        raise ValueError("invalid episode schedule")
    episode = frame_id // frames_per_episode
    return episode, scenario_names[episode % len(scenario_names)], frame_id % frames_per_episode


def process_alive(pid):
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="ascii").split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return len(stat) > 2 and stat[2] != "Z"


def sigkill_and_confirm(pid, timeout_s=1.0):
    pid = int(pid)
    os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if not process_alive(pid):
            return True
        time.sleep(0.01)
    return not process_alive(pid)


def transport_safe_clip(value, limit):
    limit = abs(float(limit))
    bits = struct.unpack("I", struct.pack("f", limit))[0]
    safe_limit = struct.unpack("f", struct.pack("I", bits))[0]
    if safe_limit > limit:
        safe_limit = struct.unpack("f", struct.pack("I", bits - 1))[0]
    return max(-safe_limit, min(float(value), safe_limit))
