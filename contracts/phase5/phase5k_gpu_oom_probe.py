#!/usr/bin/env python3
import json
import sys
import time
from pathlib import Path


def main():
    import torch

    receipt = Path(sys.argv[1]).resolve()
    blocks = []
    observed = False
    message = ""
    free_before, total = torch.cuda.mem_get_info()
    allocated = 0
    chunk = 256 * 1024 * 1024
    started = time.monotonic_ns()
    try:
        while True:
            block = torch.empty(chunk // 4, dtype=torch.float32, device="cuda")
            block.fill_(1.0)
            blocks.append(block)
            allocated += chunk
    except RuntimeError as error:
        message = str(error)
        observed = "out of memory" in message.lower()
    finally:
        blocks.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    result = {
        "schema_version": "phase5k-gpu-oom-v1",
        "observed": observed,
        "message_contains_cuda_oom": observed,
        "allocated_bytes_before_failure": allocated,
        "free_bytes_before": int(free_before),
        "free_bytes_after_release": int(free_after),
        "total_bytes": int(total),
        "duration_ms": (time.monotonic_ns() - started) / 1e6,
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result))
    if not observed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
