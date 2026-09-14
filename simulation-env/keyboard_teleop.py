#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dora-rs==0.3.13",
#     "pyarrow>=14.0.0"
# ]
# ///
import socket
import struct
import time
import pyarrow as pa
from dora import Node
CONTROL_STALE_SECONDS = 0.20
ZERO_CMD = (0.0, 0.0)

def main():
    # 1. 接入 DORA 拓扑网关
    print("[DORA receiver] Zero-copy keyboard teleoperation channel starting...")
    dora_node = Node()
    
    # 2. 物理绑定非阻塞式 UDP 监听套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 5005))
    sock.setblocking(False)  # 设为完全非阻塞，避免 socket 锁死 DORA 主线程
    last_packet_time = time.monotonic()
    zero_sent_after_stale = True
    
    print("========================================================")
    print("Keyboard teleoperation bridge receiver (DORA Node) started")
    print("Listening on UDP socket: 127.0.0.1:5005 (non-blocking)")
    print("========================================================")
    
    try:
        while True:
            # A. 尝试从 UDP 缓冲区非阻塞式读取遥控数据
            try:
                data, addr = sock.recvfrom(1024)
                if len(data) == 8:
                    # 极速解析 IEEE 754 32位连续浮点数 [v, w]
                    v, w = struct.unpack('ff', data)
                    # 严格按照 FSD 运动指令契约打包为 Arrow Float32 数组广播
                    arrow_cmd = pa.array([v, w], type=pa.float32())
                    dora_node.send_output("control_cmd", arrow_cmd)
                    last_packet_time = time.monotonic()
                    zero_sent_after_stale = False
            except BlockingIOError:
                pass  # 缓冲区无数据，平滑过渡
            if not zero_sent_after_stale and time.monotonic() - last_packet_time > CONTROL_STALE_SECONDS:
                dora_node.send_output("control_cmd", pa.array(ZERO_CMD, type=pa.float32()))
                zero_sent_after_stale = True
            
            # Poll DORA lifecycle events without blocking
            event = dora_node.next(0.002)
            if event is not None:
                if event["type"] == "STOP":
                    print("\n[DORA receiver] DORA stop signal received; shutting down safely.")
                    break
            
            # 维持 500Hz 的超高频轮询，保护 CPU 不发生空转
            time.sleep(0.002)
    except Exception as e:
        print(f"\n[DORA receiver] Runtime error: {e}")
    finally:
        sock.close()
        print("[DORA receiver] UDP socket closed; node unloaded safely.")

if __name__ == "__main__":
    main()
