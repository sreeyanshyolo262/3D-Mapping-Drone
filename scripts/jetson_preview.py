#!/usr/bin/env python3
"""Runs ON THE JETSON. Serves a live preview of the D455 (color | depth) as an MJPEG web page.

Open  http://<jetson-ip>:8080  in a browser on the laptop.
Needs the realsense2_camera node already running (jetson_preview.sh starts it).
"""
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

CW = int(sys.argv[1]) if len(sys.argv) > 1 else 640      # colour size
CH = int(sys.argv[2]) if len(sys.argv) > 2 else 480
DW = int(sys.argv[3]) if len(sys.argv) > 3 else 640      # depth size
DH = int(sys.argv[4]) if len(sys.argv) > 4 else 480
PORT = int(sys.argv[5]) if len(sys.argv) > 5 else 8080
MIN_MM, MAX_MM = 300, 4000          # depth range mapped onto the color scale

latest = {"color": None, "depth": None}
lock = threading.Lock()
jpeg = {"data": None, "seq": 0}
cond = threading.Condition()


def raw_bytes(msg):
    return msg if isinstance(msg, (bytes, bytearray)) else bytes(msg.data)


class Grabber(Node):
    def __init__(self):
        super().__init__("preview_grabber")
        for key, topic in (("color", "/camera/color/image_raw"),
                           ("depth", "/camera/depth/image_rect_raw")):
            cb = self._make_cb(key)
            try:   # raw=True skips slow Python message deserialisation
                self.create_subscription(Image, topic, cb, qos_profile_sensor_data, raw=True)
            except TypeError:
                self.create_subscription(Image, topic, cb, qos_profile_sensor_data)

    def _make_cb(self, key):
        def cb(msg):
            with lock:
                latest[key] = raw_bytes(msg)
        return cb


def build_frame(color_buf, depth_buf, fps):
    # The pixel data is the LAST field of a sensor_msgs/Image, so take it from the tail.
    if color_buf is not None and len(color_buf) >= CW * CH * 3:
        c = np.frombuffer(color_buf[-CW * CH * 3:], np.uint8).reshape(CH, CW, 3)[:, :, ::-1]
        color = np.ascontiguousarray(c)
    else:
        color = np.zeros((CH, CW, 3), np.uint8)
    centre_txt = "no depth"
    if depth_buf is not None and len(depth_buf) >= DW * DH * 2:
        d = np.frombuffer(depth_buf[-DW * DH * 2:], "<u2").reshape(DH, DW)
        s = np.clip((d.astype(np.float32) - MIN_MM) / (MAX_MM - MIN_MM), 0, 1)
        gray = (255 - s * 230).astype(np.uint8)      # grayscale depth: near = bright, far = dark
        gray[d == 0] = 0                             # no reading -> black
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        patch = d[DH // 2 - 5:DH // 2 + 6, DW // 2 - 5:DW // 2 + 6]
        valid = patch[patch > 0]
        if valid.size:
            centre_txt = "%.2f m" % (float(np.median(valid)) / 1000.0)
        else:
            centre_txt = "-- m"
    else:
        vis = np.zeros((DH, DW, 3), np.uint8)
    # Depth is shown at the colour panel's height so the two sit side by side.
    dw_show = int(round(DW * CH / DH))
    vis = cv2.resize(vis, (dw_show, CH), interpolation=cv2.INTER_NEAREST)
    img = np.hstack([color, vis])
    for x0 in (CW // 2, CW + dw_show // 2):           # centre crosshair on each panel
        cv2.drawMarker(img, (x0, CH // 2), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(img, "centre: " + centre_txt, (CW + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "color %dx%d" % (CW, CH), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "%.0f fps" % fps, (10, CH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA)
    return img


def encoder_loop():
    last = time.time()
    fps = 0.0
    prev = (None, None)
    while True:
        with lock:
            cur = (latest["color"], latest["depth"])
        if cur[0] is prev[0] and cur[1] is prev[1]:
            time.sleep(0.01)
            continue
        prev = cur
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
        last = now
        ok, buf = cv2.imencode(".jpg", build_frame(cur[0], cur[1], fps),
                               [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with cond:
                jpeg["data"] = buf.tobytes()
                jpeg["seq"] += 1
                cond.notify_all()
        time.sleep(1 / 20)


PAGE = (b"<!doctype html><title>D455 live</title>"
        b"<body style='margin:0;background:#000;color:#ccc;font-family:sans-serif'>"
        b"<div style='padding:6px 10px'>D455 live preview &mdash; left: color, right: depth "
        b"(bright = near, dark = far, black = no reading)</div>"
        b"<img src='/stream.mjpg' style='width:100%;display:block'></body>")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path.startswith("/snapshot.jpg"):
            with cond:
                cond.wait_for(lambda: jpeg["data"] is not None, timeout=10)
                data = jpeg["data"]
            if data is None:
                self.send_error(503, "no frames yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seq = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: jpeg["seq"] != seq, timeout=5)
                        data, seq = jpeg["data"], jpeg["seq"]
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    rclpy.init()
    node = Grabber()
    threading.Thread(target=encoder_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("[preview] serving on port %d  (color %dx%d, depth %dx%d)" % (PORT, CW, CH, DW, DH),
          flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
