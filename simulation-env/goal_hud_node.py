# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dora-rs==0.3.13",
#     "numpy>=1.26.0",
#     "opencv-python>=4.8.0",
#     "pyarrow>=14.0.0"
# ]
# ///
import math
import time

import cv2
import numpy as np
from dora import Node

WINDOW_NAME = "FSD Spice Goal HUD"
CANVAS_SIZE = 460
CAR_CENTER = (230, 270)
PX_PER_METER = 55.0

def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def local_goal(odom, goal):
    dx = goal[0] - odom[0]
    dy = goal[1] - odom[1]
    yaw = odom[2]
    forward = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return forward, left, math.sqrt(dx * dx + dy * dy)

def clipped_point(forward, left):
    x = CAR_CENTER[0] - left * PX_PER_METER
    y = CAR_CENTER[1] - forward * PX_PER_METER
    dx = x - CAR_CENTER[0]
    dy = y - CAR_CENTER[1]
    length = math.hypot(dx, dy)
    max_len = 175.0
    if length > max_len:
        scale = max_len / length
        x = CAR_CENTER[0] + dx * scale
        y = CAR_CENTER[1] + dy * scale
    return int(round(x)), int(round(y))

def guidance_text(bearing_rad):
    deg = math.degrees(bearing_rad)
    if abs(deg) < 10.0:
        return "HOLD STRAIGHT", (80, 220, 80)
    if deg > 0.0:
        return "TURN LEFT", (0, 210, 255)
    return "TURN RIGHT", (0, 210, 255)

def draw_hud(odom, goal, last_goal_time, collecting_active):
    canvas = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 24, dtype=np.uint8)
    cv2.circle(canvas, CAR_CENTER, int(1.0 * PX_PER_METER), (55, 55, 55), 1)
    cv2.circle(canvas, CAR_CENTER, int(2.0 * PX_PER_METER), (55, 55, 55), 1)
    cv2.circle(canvas, CAR_CENTER, int(3.0 * PX_PER_METER), (45, 45, 45), 1)
    cv2.line(canvas, (CAR_CENTER[0], 40), (CAR_CENTER[0], CANVAS_SIZE - 30), (60, 60, 60), 1)
    cv2.line(canvas, (40, CAR_CENTER[1]), (CANVAS_SIZE - 40, CAR_CENTER[1]), (60, 60, 60), 1)

    forward, left, distance = local_goal(odom, goal)
    bearing = math.atan2(left, forward)
    goal_pt = clipped_point(forward, left)

    cv2.arrowedLine(canvas, CAR_CENTER, goal_pt, (0, 220, 255), 3, tipLength=0.18)
    cv2.circle(canvas, goal_pt, 13, (0, 220, 255), -1)
    cv2.circle(canvas, goal_pt, 17, (0, 120, 255), 2)

    car = np.array([
        [CAR_CENTER[0], CAR_CENTER[1] - 22],
        [CAR_CENTER[0] - 14, CAR_CENTER[1] + 18],
        [CAR_CENTER[0] + 14, CAR_CENTER[1] + 18],
    ], dtype=np.int32)
    cv2.fillConvexPoly(canvas, car, (220, 220, 220))
    cv2.putText(canvas, "CAR", (CAR_CENTER[0] - 23, CAR_CENTER[1] + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)

    prompt, color = guidance_text(bearing)
    cv2.putText(canvas, "GOAL NAVIGATION", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
    cv2.putText(canvas, f"Distance: {distance:5.2f} m", (24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
    cv2.putText(canvas, f"Bearing : {math.degrees(bearing):+5.1f} deg", (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
    cv2.putText(canvas, f"Speed   : {odom[3]:5.2f} m/s", (24, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2)
    if collecting_active:
        cv2.putText(canvas, prompt, (24, CANVAS_SIZE - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(canvas, "MOVE CAR TO NEW START", (24, CANVAS_SIZE - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 210, 255), 2)

    stale = time.time() - last_goal_time
    if stale > 0.5:
        cv2.putText(canvas, "GOAL STALE", (CANVAS_SIZE - 160, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
    return canvas

def main():
    print("========================================================")
    print("FSD Spice Goal HUD active. Showing goal direction and distance.")
    print("========================================================")
    dora_node = Node()
    odom = [0.0, 0.0, 0.0, 0.0]
    goal = [0.52, 4.11]
    collecting_active = True
    last_goal_time = time.time()
    last_draw_time = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_SIZE, CANVAS_SIZE)

    try:
        while True:
            event = dora_node.next(timeout=0.02)
            if event is not None:
                if event["type"] == "INPUT":
                    ev_id = event["id"]
                    data = event["value"].to_numpy()
                    if ev_id == "odometry" and len(data) >= 3:
                        odom = [
                            float(data[0]),
                            float(data[1]),
                            float(data[2]),
                            float(data[3]) if len(data) >= 4 else odom[3],
                        ]
                    elif ev_id == "human_prior" and len(data) >= 2:
                        goal = [float(data[0]), float(data[1])]
                        if len(data) >= 3:
                            collecting_active = float(data[2]) > 0.5
                        last_goal_time = time.time()
                elif event["type"] == "STOP":
                    break

            now = time.time()
            if now - last_draw_time >= 0.05:
                cv2.imshow(WINDOW_NAME, draw_hud(odom, goal, last_goal_time, collecting_active))
                cv2.waitKey(1)
                last_draw_time = now
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
