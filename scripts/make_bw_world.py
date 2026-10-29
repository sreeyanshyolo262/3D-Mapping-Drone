"""Convert an RTAB-Map point cloud (.ply) into a black-and-white 3D world.

Outputs (next to the input):
  <name>_bw.ply    grayscale point cloud (open in MeshLab / CloudCompare)
  <name>_bw.html   interactive viewer: orbit/zoom/pan + click-to-measure distances (meters)

Usage:  python make_bw_world.py path\\to\\map_cloud.ply [max_points]
"""
import base64
import sys
from pathlib import Path

import numpy as np

NUMPY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def read_ply(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise SystemExit("Only binary_little_endian PLY is supported")
        n = next(int(h.split()[-1]) for h in header if h.startswith("element vertex"))
        fields = []
        in_vertex = False
        for h in header:
            if h.startswith("element "):
                in_vertex = h.startswith("element vertex")
            elif h.startswith("property ") and in_vertex:
                _, t, name = h.split()
                fields.append((name, NUMPY_TYPES[t]))
        data = np.fromfile(f, dtype=np.dtype(fields), count=n)
    return data


def main():
    src = Path(sys.argv[1])
    max_points = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
    bw = "bw" in sys.argv[3:]                       # optional: black-and-white shaded look
    tag = "_bw" if bw else "_world"
    d = read_ply(src)
    names = d.dtype.names
    xyz = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float32)
    print(f"read {len(xyz):,} points, fields: {names}")

    # Light direction from the camera-ish side so surfaces facing us are bright.
    if all(k in names for k in ("nx", "ny", "nz")):
        nrm = np.stack([d["nx"], d["ny"], d["nz"]], axis=1).astype(np.float32)
        light = np.array([0.35, -0.5, 0.8], np.float32)
        light /= np.linalg.norm(light)
        shade = np.abs(nrm @ light)
    else:
        shade = np.ones(len(xyz), np.float32)

    # Height cue: higher points slightly brighter, so floor vs. ceiling separates.
    z = xyz[:, 2]
    lo, hi = np.percentile(z, [1, 99])
    h = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)

    gray = np.clip(0.25 + 0.55 * shade + 0.20 * h, 0, 1)
    g8 = (gray * 255).astype(np.uint8)
    if not bw and all(k in names for k in ("red", "green", "blue")):
        # Keep the camera's real colours, like the RTAB-Map viewer does.
        rgb = np.stack([d["red"], d["green"], d["blue"]], axis=1).astype(np.uint8)
    else:
        rgb = np.stack([g8, g8, g8], axis=1)

    # Drop isolated floating specks: keep points whose coarse voxel has neighbours.
    vox = np.floor(xyz / 0.04).astype(np.int64)
    key = (vox[:, 0] * 73856093) ^ (vox[:, 1] * 19349663) ^ (vox[:, 2] * 83492791)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 3
    xyz, rgb = xyz[keep], rgb[keep]
    print(f"after speck filter: {len(xyz):,}")

    # PLY (full density)
    ply = src.with_name(src.stem.replace("_cloud", "") + tag + ".ply")
    out = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["red"], out["green"], out["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(ply, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(out)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode("ascii"))
        out.tofile(f)
    print("wrote", ply)

    # Interactive viewer (downsampled)
    if len(xyz) > max_points:
        idx = np.random.default_rng(1).choice(len(xyz), max_points, replace=False)
        vx, vg = xyz[idx], rgb[idx]
    else:
        vx, vg = xyz, rgb
    centre = np.median(vx, axis=0)
    vx = vx - centre
    # RTAB-Map uses ROS axes (x fwd, y left, z up); three.js is y-up.
    three = np.stack([vx[:, 0], vx[:, 2], -vx[:, 1]], axis=1).astype(np.float32)
    pos_b64 = base64.b64encode(three.tobytes()).decode()
    gry_b64 = base64.b64encode(np.ascontiguousarray(vg).tobytes()).decode()
    extent = float(np.percentile(np.linalg.norm(three, axis=1), 98))
    # Where the scan started (map origin), in viewer coordinates: the first-person start view.
    start = (-centre[0], -centre[2], centre[1])

    html = HTML.replace("__POS__", pos_b64).replace("__GRY__", gry_b64) \
               .replace("__N__", str(len(three))).replace("__EXTENT__", f"{extent:.3f}") \
               .replace("__START__", "%.3f,%.3f,%.3f" % start)
    page = src.with_name(src.stem.replace("_cloud", "") + tag + ".html")
    page.write_text(html, encoding="utf-8")
    print("wrote", page, f"({len(three):,} points in viewer)")


HTML = (Path(__file__).with_name("viewer_template.html")).read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
