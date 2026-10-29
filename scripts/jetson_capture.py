#!/usr/bin/env python3
"""Runs ON THE JETSON. Grabs ONE colour frame and the matching depth frame (aligned to the
colour image, so pixel (u,v) means the same thing in both) from the running D455 and writes:

  <prefix>_color.png        normal camera view
  <prefix>_depth_mm.npy     depth in millimetres, uint16, same size as the colour image (0 = no reading)
  <prefix>_depth.png        grayscale depth, near = bright, holes black
  <prefix>_depth_filled.png the same with holes filled
  <prefix>_intrinsics.json  fx, fy, cx, cy, width, height (needed to measure distances)

Usage: python3 jetson_capture.py <prefix> [warmup_seconds]
"""
import json
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "capture"
WARMUP = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
MIN_MM, MAX_MM = 300, 4000


class Grab(Node):
    def __init__(self):
        super().__init__("capture_once")
        self.color = None
        self.depth = None
        self.info = None
        q = qos_profile_sensor_data
        self.create_subscription(Image, "/camera/color/image_raw", self.on_color, q)
        self.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", self.on_depth, q)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.on_info, q)

    def on_color(self, m):
        self.color = m

    def on_depth(self, m):
        self.depth = m

    def on_info(self, m):
        self.info = m


def main():
    rclpy.init()
    node = Grab()
    t0 = time.time()
    # Let auto-exposure settle, then take the latest frames.
    while time.time() - t0 < WARMUP + 30:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (time.time() - t0 > WARMUP and node.color is not None
                and node.depth is not None and node.info is not None):
            break
    if node.color is None or node.depth is None or node.info is None:
        print("no frames received (color=%s aligned_depth=%s camera_info=%s)"
              % (node.color is not None, node.depth is not None, node.info is not None))
        sys.exit(1)

    c = node.color
    raw = np.frombuffer(bytes(c.data), np.uint8)
    if c.encoding == "rgb8":
        img = cv2.cvtColor(raw.reshape(c.height, c.width, 3), cv2.COLOR_RGB2BGR)
    elif c.encoding == "bgr8":
        img = raw.reshape(c.height, c.width, 3)
    else:
        print("unexpected colour encoding:", c.encoding)
        sys.exit(1)
    cv2.imwrite(PREFIX + "_color.png", img)

    d = node.depth
    depth = np.frombuffer(bytes(d.data), "<u2").reshape(d.height, d.width)
    if depth.shape != img.shape[:2]:
        print("aligned depth %s does not match colour %s" % (depth.shape, img.shape[:2]))
        sys.exit(1)
    np.save(PREFIX + "_depth_mm.npy", depth)

    k = node.info.k
    with open(PREFIX + "_intrinsics.json", "w") as f:
        json.dump({"width": int(c.width), "height": int(c.height),
                   "fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5]}, f)

    s = np.clip((depth.astype(np.float32) - MIN_MM) / (MAX_MM - MIN_MM), 0, 1)
    gray = (255 - s * 230).astype(np.uint8)          # near bright, far dark
    gray[depth == 0] = 0
    cv2.imwrite(PREFIX + "_depth.png", gray)
    holes = (depth == 0).astype(np.uint8) * 255
    cv2.imwrite(PREFIX + "_depth_filled.png", cv2.inpaint(gray, holes, 5, cv2.INPAINT_TELEA))

    valid = depth[depth > 0]
    print("colour %dx%d (%s), valid depth %.0f%%, range %.2f-%.2f m, fx=%.1f"
          % (c.width, c.height, c.encoding, 100.0 * valid.size / depth.size,
             valid.min() / 1000.0 if valid.size else 0,
             valid.max() / 1000.0 if valid.size else 0, k[0]))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
