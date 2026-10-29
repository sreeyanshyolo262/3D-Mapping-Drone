"""Render an RTAB-Map point cloud (.ply) as a dense grayscale depth map (near = bright).

Looks at the cloud from a virtual camera (default: the map origin, looking along the first
camera direction), keeps the nearest point per pixel, fills the gaps between points and
smooths the result, so you get a continuous surface image instead of scattered dots.

Usage:
  python make_depth_map.py path\\to\\map_cloud.ply
  python make_depth_map.py map_cloud.ply --yaw 25 --pitch -5        # look a bit left / down
  python make_depth_map.py map_cloud.ply --pos 0.5 0 0 --fov 90     # move the virtual camera

Axes are the ROS ones used by RTAB-Map: x forward, y left, z up.
Writes <name>_depth.png (grayscale) and <name>_depth_color.png next to the input.
"""
import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

from make_bw_world import read_ply


def write_png(path, img):
    """Minimal PNG writer (uint8, gray HxW or RGB HxWx3) so no imaging library is needed."""
    h, w = img.shape[:2]
    ctype = 0 if img.ndim == 2 else 2
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def shifted(a, dy, dx, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def fill_holes(depth, iters):
    """Fill empty pixels (inf) with the mean of nearby valid pixels that sit on the same
    surface (within 15% of the nearest neighbour), so gaps close without blurring edges."""
    d = depth.copy()
    offs = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    for _ in range(iters):
        holes = ~np.isfinite(d)
        if not holes.any():
            break
        stack = np.stack([shifted(d, dy, dx, np.inf) for dy, dx in offs])
        nearest = stack.min(axis=0)
        ok = np.isfinite(stack) & (stack <= nearest * 1.15)
        mean = np.where(ok, stack, 0).sum(axis=0) / np.maximum(ok.sum(axis=0), 1)
        fillable = holes & np.isfinite(nearest)
        d[fillable] = mean[fillable]
    return d


def smooth(depth):
    """3x3 mean over valid pixels only (keeps the black background untouched)."""
    valid = np.isfinite(depth)
    v = np.where(valid, depth, 0.0)
    num = np.zeros_like(v)
    cnt = np.zeros_like(v)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            num += shifted(v, dy, dx, 0.0)
            cnt += shifted(valid.astype(np.float32), dy, dx, 0.0)
    out = np.where(valid, num / np.maximum(cnt, 1), np.inf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=87.0, help="horizontal field of view, degrees")
    ap.add_argument("--yaw", type=float, default=0.0, help="turn left (+) / right (-), degrees")
    ap.add_argument("--pitch", type=float, default=0.0, help="look up (+) / down (-), degrees")
    ap.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"))
    ap.add_argument("--near", type=float, default=0.2)
    ap.add_argument("--fill", type=int, default=6, help="hole-filling passes (more = smoother)")
    ap.add_argument("--splat", type=int, default=2, help="near-point growth passes (less see-through)")
    a = ap.parse_args()

    src = Path(a.ply)
    d = read_ply(src)
    p = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64) - np.array(a.pos)
    print(f"read {len(p):,} points")

    # Virtual camera rotation (yaw about z, then pitch about the camera's left axis).
    yw, pt = np.radians(a.yaw), np.radians(a.pitch)
    rz = np.array([[np.cos(yw), np.sin(yw), 0], [-np.sin(yw), np.cos(yw), 0], [0, 0, 1]])
    ry = np.array([[np.cos(pt), 0, np.sin(pt)], [0, 1, 0], [-np.sin(pt), 0, np.cos(pt)]])
    p = p @ (ry @ rz).T
    # ROS (x fwd, y left, z up) -> pinhole camera (X right, Y down, Z forward).
    z, x, y = p[:, 0], -p[:, 1], -p[:, 2]
    keep = z > a.near
    x, y, z = x[keep], y[keep], z[keep]

    W, H = a.width, a.height
    f = (W / 2) / np.tan(np.radians(a.fov) / 2)
    u = np.round(f * x / z + W / 2).astype(np.int64)
    v = np.round(f * y / z + H / 2).astype(np.int64)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inside], v[inside], z[inside]
    print(f"{len(z):,} points in view")
    if len(z) == 0:
        raise SystemExit("Nothing in view - try different --yaw / --pitch / --pos.")

    # Z-buffer: nearest point per pixel.
    key = v * W + u
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    depth = np.full(W * H, np.inf, np.float32)
    depth[key[first]] = z[first]
    depth = depth.reshape(H, W)

    # Splat: let near points cover their neighbours so far points can't show through the gaps.
    for _ in range(a.splat):
        depth = np.min([shifted(depth, dy, dx, np.inf)
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)

    depth = fill_holes(depth, a.fill)
    depth = smooth(depth)
    depth = smooth(depth)

    valid = np.isfinite(depth)
    lo, hi = np.percentile(depth[valid], [1, 99])
    t = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    t = 1.0 - t ** 0.8                     # near = bright
    gray = np.where(valid, 25 + 230 * t, 0).astype(np.uint8)

    out = src.with_name(src.stem.replace("_cloud", "") + "_depth.png")
    write_png(out, gray)
    print("wrote", out)

    # Colour version (blue = near ... red = far) using a simple ramp, no extra libraries.
    c = np.clip(1.0 - t, 0, 1)
    r = np.clip(1.5 - np.abs(4 * c - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * c - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * c - 1), 0, 1)
    rgb = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
    rgb[~valid] = 0
    out2 = src.with_name(src.stem.replace("_cloud", "") + "_depth_color.png")
    write_png(out2, rgb)
    print("wrote", out2)
    print(f"depth range shown: {lo:.2f} m (bright) to {hi:.2f} m (dark)")


if __name__ == "__main__":
    main()
