"""执行器信任边界上的控制命令校验与新鲜度判定。

这些函数是纯函数,不依赖 Isaac、Dora 或 numpy,便于在无仿真环境下单测。
数值包线取自 ``contracts/v1/data_contracts.json`` 中三条 ``control_cmd``
契约 (autonomous / teleop / bc_direct) 的并集,作为落地执行器侧的硬安全包线:

* linear_velocity  v: [0.0, 0.8] m/s
* yaw_rate         w: [-1.0, 1.0] rad/s
* 新鲜度: 200 ms 内没有有效命令即归零 (fail-closed)。

任何非有限、越界或长度不符的命令一律判为无效,调用方据此输出零速且不刷新
新鲜度时间戳,从而在源头持续异常时保持停车。
"""
from __future__ import annotations

import math

CONTROL_V_MIN = 0.0
CONTROL_V_MAX = 0.8
CONTROL_W_MIN = -1.0
CONTROL_W_MAX = 1.0
CONTROL_STALE_MS = 200.0


def parse_control_command(values):
    """校验 ``control_cmd`` 载荷。

    合法 (长度为 2、两分量均有限且落在包线内) 时返回 ``(v, w)`` 浮点元组;
    否则返回 ``None``,由调用方输出零速并保持 fail-closed。
    """
    if values is None or len(values) != 2:
        return None
    try:
        v = float(values[0])
        w = float(values[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(v) and math.isfinite(w)):
        return None
    if not (CONTROL_V_MIN <= v <= CONTROL_V_MAX):
        return None
    if not (CONTROL_W_MIN <= w <= CONTROL_W_MAX):
        return None
    return (v, w)


def is_control_fresh(last_valid_monotonic, now_monotonic, max_age_ms=CONTROL_STALE_MS):
    """基于单调时钟判定最近一条有效命令是否仍然新鲜。

    ``last_valid_monotonic`` 为 ``None`` (从未收到有效命令) 时判为不新鲜。
    """
    if last_valid_monotonic is None:
        return False
    return (now_monotonic - last_valid_monotonic) * 1000.0 <= max_age_ms
