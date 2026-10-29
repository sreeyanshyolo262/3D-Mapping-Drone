#!/usr/bin/env python3
"""Runs ON THE JETSON. Stop-and-snap capture for walk-through maps.

At each stop it grabs a short burst of frames, keeps the SHARPEST colour image and a de-noised
depth image (median of the burst), checks the snap is usable, and saves it:
    <session>/NNN_color.png   <session>/NNN_depth_mm.npy   <session>/intrinsics.json   <session>/snaps.json

Usage: python3 jetson_snap.py <session_dir> [auto_seconds]
  auto_seconds = 0 (default): press Enter to snap, type q + Enter to finish.
  auto_seconds > 0          : snaps by itself every N seconds (Enter still snaps right now).
                              Ctrl+C or q + Enter finishes.
Needs the realsense2_camera node running with align_depth.enable:=true (jetson_snap.sh does it).
"""
import json
import select
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "snaps"
AUTO = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
BURST = 8                       # frames looked at per snap
MIN_MM, MAX_MM = 300, 4000      # depth range that is trusted
MIN_VALID = 0.35                # fraction of pixels that must have usable depth
BLUR_RATIO = 0.45               # reject if sharpness < this x the median of accepted snaps


class Grab(Node):
    def __init__(self):
        super().__init__("snap_grabber")
        self.lock = threading.Lock()
        self.color = self.depth = self.info = None
        self.cseq = self.dseq = 0
        q = qos_profile_sensor_data
        self.create_subscription(Image, "/camera/color/image_raw", self.on_color, q)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.on_depth, q)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, q)

    def on_color(self, m):
        a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        a = a[:, :, ::-1] if m.encoding == "rgb8" else a        # -> BGR
        with self.lock:
            self.color, self.cseq = np.ascontiguousarray(a), self.cseq + 1

    def on_depth(self, m):
        a = np.frombuffer(bytes(m.data), "<u2").reshape(m.height, m.width)
        with self.lock:
            self.depth, self.dseq = a.copy(), self.dseq + 1

    def on_info(self, m):
        with self.lock:
            self.info = m


def burst(node):
    """Collect up to BURST (sharpness, colour, depth) triples."""
    frames, last, t0 = [], (-1, -1), time.time()
    while len(frames) < BURST and time.time() - t0 < 5.0:
        with node.lock:
            c, d, key = node.color, node.depth, (node.cseq, node.dseq)
        if c is None or d is None or key == last:
            time.sleep(0.02)
            continue
        last = key
        sharp = cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        frames.append((sharp, c, d))
    return frames


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    rclpy.init()
    node = Grab()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("[snap] waiting for the camera ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 60:
        with node.lock:
            if node.color is not None and node.depth is not None and node.info is not None:
                break
        time.sleep(0.2)
    else:
        print("[snap] ERROR: no camera frames after 60 s", flush=True)
        sys.exit(1)
    k = node.info.k
    h, w = node.color.shape[:2]
    json.dump({"width": w, "height": h, "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5]},
              open(os.path.join(OUT, "intrinsics.json"), "w"))

    snaps, sharps = [], []

    def take():
        n = len(snaps)
        frames = burst(node)
        if len(frames) < 3:
            print("[snap] RETAKE: not enough camera frames arrived", flush=True)
            return
        best = max(f[0] for f in frames)
        sharp, color, _ = max(frames, key=lambda f: f[0])
        still = [f[2] for f in frames if f[0] >= 0.7 * best]       # frames from when you were still
        st = np.stack(still).astype(np.float32)
        st[st == 0] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(st, axis=0)
        depth = np.nan_to_num(med, nan=0.0).astype(np.uint16)
        valid = float(((depth >= MIN_MM) & (depth <= MAX_MM)).mean())

        why = []
        if len(sharps) >= 3 and sharp < BLUR_RATIO * float(np.median(sharps)):
            why.append("blurry / moving - hold still")
        if valid < MIN_VALID:
            why.append("little depth (%.0f%%) - too close/far, or window/shiny surface" % (100 * valid))
        if why:
            print("[snap] RETAKE: " + "; ".join(why), flush=True)
            return

        name = "%03d" % n
        cv2.imwrite(os.path.join(OUT, name + "_color.png"), color)
        np.save(os.path.join(OUT, name + "_depth_mm.npy"), depth)
        sharps.append(sharp)
        snaps.append({"index": n, "name": name, "sharp": round(sharp, 1),
                      "valid": round(valid, 3), "time": time.time()})
        json.dump(snaps, open(os.path.join(OUT, "snaps.json"), "w"))
        print("[snap] saved #%d  sharp %.0f  depth %.0f%%  OK" % (n + 1, sharp, 100 * valid), flush=True)

    if AUTO > 0:
        print("[snap] AUTO: a snap every %.0f s. HOLD STILL when it says 'snapping'. "
              "Enter = snap now, q + Enter or Ctrl+C = finish." % AUTO, flush=True)
    else:
        print("[snap] READY. Hold still, press Enter to snap. q + Enter = finish.", flush=True)

    try:
        nxt, shown = time.time() + AUTO, -1
        while True:
            if AUTO > 0:
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    s = sys.stdin.readline().strip().lower()
                    if s in ("q", "quit", "exit"):
                        break
                    take()
                    nxt = time.time() + AUTO
                    continue
                left = nxt - time.time()
                if left <= 0:
                    print("[snap] snapping ...", flush=True)
                    take()
                    nxt, shown = time.time() + AUTO, -1
                elif left <= 3 and int(left) + 1 != shown:
                    shown = int(left) + 1
                    print("[snap] %d ..." % shown, flush=True)
            else:
                s = input("> ").strip().lower()
                if s in ("q", "quit", "exit"):
                    break
                take()
    except (KeyboardInterrupt, EOFError):
        pass
    print("[snap] finished: %d snaps saved in %s" % (len(snaps), OUT), flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
