"""Join stop-and-snap photos (colour + depth) into one 3D map you can walk through like Street View.

Usage:  python make_streetview.py path\\to\\snaps_YYYYMMDD_HHMMSS [--no-loops] [--step 2] [--max-points 2500000]

How it works
  1. Each snap's position is found relative to the previous one by matching image features (SIFT),
     solving the camera move from the depth (PnP + RANSAC) and double-checking it against the
     depth image, so wrong matches (repeating desks!) are rejected.
  2. If you come back near an earlier snap, the drift is corrected along the way ("loop closure").
  3. Every snap's depth is turned into coloured 3D points, merged, and de-duplicated.
  4. A web page is written: click the dots / use the arrow keys to step between your snap spots,
     drag to look around, Tab to fly freely around the whole map.

Writes  <session>\\streetview.html  and  <session>\\world.ply
"""
import argparse
import base64
import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])        # camera optical frame (x right, y down, z fwd) -> three.js (y up)
MIN_Z, MAX_Z = 0.3, 4.0                  # trusted depth range, metres


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def keypoint_depth(depth, pts, win=2):
    """Depth at each feature: median of a small window, and rejected where the window straddles
    a depth edge (object borders mix foreground and background depth and ruin the 3D position)."""
    h, w = depth.shape
    if len(pts) == 0:
        return np.zeros(0, np.float32), np.zeros(0, bool)
    ui = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    pad = np.pad(depth, win, mode="constant")
    patch = np.stack([pad[win + dy + vi, win + dx + ui]
                      for dy in range(-win, win + 1) for dx in range(-win, win + 1)]).astype(np.float64)
    patch[patch <= 0] = np.nan
    with np.errstate(all="ignore"):
        med = np.nanmedian(patch, axis=0)
        spread = np.nanmax(patch, axis=0) - np.nanmin(patch, axis=0)
    cnt = np.sum(~np.isnan(patch), axis=0)
    ok = (cnt >= 10) & (spread < 0.04 * med + 0.03) & (med > MIN_Z) & (med < MAX_Z)
    return np.where(ok, med, 0.0).astype(np.float32), ok


class Snap:
    def __init__(self, name, color, depth_m, sift):
        self.name, self.color, self.depth = name, color, depth_m
        self.h, self.w = depth_m.shape
        kp, self.des = sift.detectAndCompute(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), None)
        self.pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        self.z, self.zok = keypoint_depth(depth_m, self.pts)


def depth_agreement(a, b, T, K, step=6):
    """Fraction of a's depth pixels that land on matching depth in b under the move T (X_b = T X_a)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = a.depth[::step, ::step]
    vs, us = np.mgrid[0:a.h:step, 0:a.w:step]
    m = (d > MIN_Z) & (d < MAX_Z)
    z = d[m]
    X = np.stack([(us[m] - cx) * z / fx, (vs[m] - cy) * z / fy, z], 1)
    Xb = X @ T[:3, :3].T + T[:3, 3]
    ok = Xb[:, 2] > 0.2
    ub = np.where(ok, Xb[:, 0] / np.where(ok, Xb[:, 2], 1) * fx + cx, -1)
    vb = np.where(ok, Xb[:, 1] / np.where(ok, Xb[:, 2], 1) * fy + cy, -1)
    ok &= (ub >= 0) & (ub < b.w - 1) & (vb >= 0) & (vb < b.h - 1)
    zb = np.zeros(len(z), np.float32)
    zb[ok] = b.depth[np.round(vb[ok]).astype(int), np.round(ub[ok]).astype(int)]
    has = ok & (zb > MIN_Z)
    n = int(has.sum())
    if n == 0:
        return 0.0, 0
    err = np.abs(Xb[has, 2] - zb[has])
    return float((err <= 0.04 + 0.03 * zb[has]).mean()), n


BF = cv2.BFMatcher(cv2.NORM_L2)


def _kabsch(P, Q):
    """Rotation R minimising |Q - R P| for matching direction vectors (rows)."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _rotation_ransac(b1, b2, thresh_deg=1.5, iters=300):
    """Rotation between two views from matched viewing directions (needs no depth at all)."""
    n = len(b1)
    rng = np.random.default_rng(0)
    cos_t = math.cos(math.radians(thresh_deg))
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R = _kabsch(b1[idx], b2[idx])
        inl = np.sum((b1 @ R.T) * b2, axis=1) > cos_t
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 8:
        return None, None
    R = _kabsch(b1[best], b2[best])
    inl = np.sum((b1 @ R.T) * b2, axis=1) > cos_t
    return R, inl


def _backproject(snap, pts, K):
    z = snap.depth[np.clip(np.round(pts[:, 1]).astype(int), 0, snap.h - 1),
                   np.clip(np.round(pts[:, 0]).astype(int), 0, snap.w - 1)].astype(np.float64)
    return np.stack([(pts[:, 0] - K[0, 2]) * z / K[0, 0], (pts[:, 1] - K[1, 2]) * z / K[1, 1], z], 1)


def icp_refine(a, b, M, K, iters=25, step=4):
    """Nudge the move M (X_b = M X_a) until the two depth images line up. Each round matches
    every depth pixel of a to the depth pixel of b it lands on, then refits the rigid move."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = a.depth[::step, ::step]
    vs, us = np.mgrid[0:a.h:step, 0:a.w:step]
    m = (d > MIN_Z) & (d < MAX_Z)
    z = d[m].astype(np.float64)
    XA = np.stack([(us[m] - cx) * z / fx, (vs[m] - cy) * z / fy, z], 1)
    gate = 0.30
    M = M.copy()
    for _ in range(iters):
        Xb = XA @ M[:3, :3].T + M[:3, 3]
        front = Xb[:, 2] > 0.2
        zz = np.where(front, Xb[:, 2], 1.0)
        ub = np.round(Xb[:, 0] / zz * fx + cx).astype(int)
        vb = np.round(Xb[:, 1] / zz * fy + cy).astype(int)
        inside = front & (ub >= 0) & (ub < b.w) & (vb >= 0) & (vb < b.h)
        zb = np.zeros(len(XA))
        zb[inside] = b.depth[vb[inside], ub[inside]]
        has = inside & (zb > MIN_Z) & (zb < MAX_Z)
        Xo = np.stack([(ub - cx) * zb / fx, (vb - cy) * zb / fy, zb], 1)
        sel = has & (np.linalg.norm(Xo - Xb, axis=1) < gate)
        if sel.sum() < 300:
            break
        Pp, Qq = Xb[sel], Xo[sel]
        cp, cq = Pp.mean(0), Qq.mean(0)
        U, _, Vt = np.linalg.svd((Pp - cp).T @ (Qq - cq))
        Rd = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = Rd, cq - Rd @ cp
        M = D @ M
        gate = max(0.06, gate * 0.8)
    return M


def _epipolar_direction(b1, b2, R):
    """Direction of the camera move given the rotation: every match must satisfy
    (R b1 x b2) . t = 0, so t is the direction those constraints leave free."""
    N = np.cross(b1 @ R.T, b2)
    _, _, Vt = np.linalg.svd(N, full_matrices=False)
    return Vt[-1]


def register(a, b, K, min_inl=20):
    """Camera move between two snaps: (T_b_a, matches used, depth agreement) or None.

    Candidate moves are built three ways and the one that the dense depth images agree with
    best is kept, so a wrong match (repeating desks, plain walls) is rejected:
      1. rotation from ALL matches (no depth needed) + sideways shift from depth where it exists
      2. the same, refined with a perspective-n-point fit
      3. classic depth-based perspective-n-point fit
    Each candidate is also polished by lining up the two depth images (ICP).
    """
    if a.des is None or b.des is None or len(a.pts) < 30 or len(b.pts) < 30:
        return None
    knn = BF.knnMatch(a.des, b.des, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < min_inl:
        return None
    qi = np.array([g.queryIdx for g in good])
    ti = np.array([g.trainIdx for g in good])
    pa, pb = a.pts[qi].astype(np.float64), b.pts[ti].astype(np.float64)

    def ray(p):
        r = np.stack([(p[:, 0] - K[0, 2]) / K[0, 0], (p[:, 1] - K[1, 2]) / K[1, 1],
                      np.ones(len(p))], 1)
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    cands = []

    R, inl = _rotation_ransac(ray(pa), ray(pb))
    n_rot = 0 if inl is None else int(inl.sum())
    if R is not None and n_rot >= min_inl:
        both = inl & a.zok[qi] & b.zok[ti]
        T = np.zeros(3)
        if both.sum() >= 6:
            XA, XB = _backproject(a, pa[both], K), _backproject(b, pb[both], K)
            D = XB - XA @ R.T
            T = np.median(D, axis=0)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R, T
        cands.append((M, n_rot, 2))
        # refine with a perspective-n-point fit on matches whose 3D position is known
        use = inl & a.zok[qi]
        if use.sum() >= 8:
            XA = _backproject(a, pa[use], K)
            rv, tv = cv2.solvePnPRefineLM(XA, pb[use], K, None, cv2.Rodrigues(R)[0],
                                          T.reshape(3, 1).copy())
            M2 = np.eye(4)
            M2[:3, :3], M2[:3, 3] = cv2.Rodrigues(rv)[0], tv[:, 0]
            cands.append((M2, n_rot, 1))

    # step-length search: direction from the matches, length from the depth images
    if R is not None and n_rot >= min_inl:
        t_dir = _epipolar_direction(ray(pa)[inl], ray(pb)[inl], R)
        scored = []
        for sgn in (1.0, -1.0):
            for dist in np.arange(0.1, 2.01, 0.1):
                M = np.eye(4)
                M[:3, :3], M[:3, 3] = R, sgn * dist * t_dir
                ag, n = depth_agreement(a, b, M, K)
                if n >= 500:
                    scored.append((ag, M))
        for ag, M in sorted(scored, key=lambda x: -x[0])[:3]:
            cands.append((M, n_rot, 3))

    # classic depth-based fit
    dg = a.zok[qi]
    if dg.sum() >= 8:
        P = _backproject(a, pa[dg], K)
        ok, rvec, tvec, inl3 = cv2.solvePnPRansac(P, pb[dg], K, None, iterationsCount=400,
                                                  reprojectionError=4.0, confidence=0.999,
                                                  flags=cv2.SOLVEPNP_EPNP)
        if ok and inl3 is not None and len(inl3) >= 8:
            inl3 = inl3[:, 0]
            rvec, tvec = cv2.solvePnPRefineLM(P[inl3], pb[dg][inl3], K, None, rvec, tvec)
            M3 = np.eye(4)
            M3[:3, :3], M3[:3, 3] = cv2.Rodrigues(rvec)[0], tvec[:, 0]
            cands.append((M3, len(inl3), 0))

    scored = []
    for M0, cnt, prio in cands:
        for M, pr in ((M0, prio), (icp_refine(a, b, M0, K), prio + 10)):
            agree, n = depth_agreement(a, b, M, K)
            if n >= 300:
                scored.append((agree, pr, M, cnt, n))
    if not scored:
        return None
    top = max(x[0] for x in scored)
    # Flat walls let many positions score alike on depth, so among near-equal scores trust the
    # candidate built from the most direct evidence (lowest priority number).
    agree, _, M, cnt, n = min((x for x in scored if x[0] >= top - 0.05), key=lambda x: x[1])
    return (M, cnt, agree, n)


def link_quality(r):
    """How far to trust a registration: 'good', 'ok' (well supported, looser depth check),
    'approx' (best effort, flagged on the page), or None (no usable evidence)."""
    if r is None:
        return None
    _, cnt, agree, n = r
    if n >= 500 and agree >= 0.7:
        return "good"
    if n >= 500 and cnt >= 30 and agree >= 0.5:
        return "ok"
    if cnt >= 25 and agree >= 0.25:
        return "approx"
    return None


def build_chain(snaps, K, log):
    n = len(snaps)
    poses, island, approx = [np.eye(4)] + [None] * (n - 1), [0] * n, 0
    isl = 0
    for i in range(1, n):
        found, weak = None, None
        for j in range(i - 1, max(-1, i - 4), -1):
            r = register(snaps[j], snaps[i], K)
            q = link_quality(r)
            if q in ("good", "ok"):
                found = (j, r, q)
                break
            if q == "approx" and (weak is None or r[2] > weak[1][2]):
                weak = (j, r, q)
        found = found or weak
        if found:
            j, (T_ba, cnt, agree, _), q = found
            poses[i], island[i] = poses[j] @ inv(T_ba), island[j]
            if q == "approx":
                approx += 1
            log("  snap %2d -> %2d %s  (%d matches, depth agreement %.0f%%)"
                % (j + 1, i + 1, "joined APPROXIMATELY" if q == "approx" else "joined", cnt, 100 * agree))
        else:
            isl += 1
            poses[i], island[i] = np.eye(4), isl
            poses[i][0, 3] = 4.0 * isl
            log("  snap %2d could NOT be joined to snaps %s  -> placed in a separate block (#%d)"
                % (i + 1, ", ".join(str(k + 1) for k in range(max(0, i - 3), i)), isl))
    return poses, island, approx


def close_loops(snaps, poses, island, K, log, max_loops=6):
    """If a snap revisits an earlier spot, spread the accumulated drift over the path between them."""
    n, done = len(poses), 0
    for i in range(n - 1, 9, -1):
        if done >= max_loops:
            break
        cands = []
        for j in range(0, i - 9):
            if island[j] != island[i]:
                continue
            dist = np.linalg.norm(poses[i][:3, 3] - poses[j][:3, 3])
            ang = math.degrees(math.acos(np.clip(poses[i][:3, 2] @ poses[j][:3, 2], -1, 1)))
            if dist < 2.5 and ang < 50:
                cands.append((dist, j))
        for dist, j in sorted(cands)[:2]:
            r = register(snaps[j], snaps[i], K, min_inl=60)
            if link_quality(r) != "good":
                continue
            T_ba, ninl = r[0], r[1]
            Pj = poses[j]
            Rel = inv(Pj) @ poses[i]
            E = inv(T_ba) @ inv(Rel)                       # drift, seen from snap j
            err = float(np.linalg.norm(E[:3, 3]))
            if err < 0.05:
                break                                      # already consistent
            if err > 3.0:
                continue                                   # implausible: likely a false match
            rv = cv2.Rodrigues(E[:3, :3])[0]
            for k in range(j + 1, n):
                f = min(1.0, (k - j) / (i - j))
                Ef = np.eye(4)
                Ef[:3, :3] = cv2.Rodrigues(rv * f)[0]
                Ef[:3, 3] = E[:3, 3] * f
                poses[k] = Pj @ Ef @ inv(Pj) @ poses[k]
            done += 1
            log("  loop closed: snap %d sees the place of snap %d again (%d matches); "
                "corrected %.2f m of drift" % (i + 1, j + 1, ninl, err))
            break
    return done


def make_cloud(snaps, poses, K, step, voxel):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P, C, Z = [], [], []
    for s, T in zip(snaps, poses):
        d = s.depth[::step, ::step]
        vs, us = np.mgrid[0:s.h:step, 0:s.w:step]
        valid = (d > MIN_Z) & (d < MAX_Z)
        jump = np.zeros_like(valid)                        # drop "flying" pixels at depth edges
        for ax in (0, 1):
            a_ = np.abs(np.diff(d, axis=ax))
            lim = 0.05 * (d[:, :-1] if ax else d[:-1])
            bad = a_ > lim
            if ax:
                jump[:, :-1] |= bad
                jump[:, 1:] |= bad
            else:
                jump[:-1] |= bad
                jump[1:] |= bad
        valid &= ~jump
        z = d[valid].astype(np.float64)
        Xc = np.stack([(us[valid] - cx) * z / fx, (vs[valid] - cy) * z / fy, z], 1)
        Xw = (Xc @ T[:3, :3].T + T[:3, 3]) @ FLIP.T
        P.append(Xw.astype(np.float32))
        C.append(s.color[::step, ::step][valid][:, ::-1])
        Z.append(z.astype(np.float32))
    P, C, Z = np.concatenate(P), np.concatenate(C), np.concatenate(Z)
    order = np.argsort(Z)                                   # keep the nearest (most accurate) observation
    P, C = P[order], C[order]
    vox = np.floor(P / voxel).astype(np.int64)
    key = (vox[:, 0] * 73856093) ^ (vox[:, 1] * 19349663) ^ (vox[:, 2] * 83492791)
    _, first = np.unique(key, return_index=True)
    return P[first], np.ascontiguousarray(C[first])


def make_spots(poses, island, P):
    """Group snaps taken from (nearly) the same place into one walk-through spot."""
    stations = []
    for T in poses:
        p = FLIP @ T[:3, 3]
        f = FLIP @ T[:3, 2]
        stations.append((p, math.atan2(-f[0], -f[2]), math.asin(float(np.clip(f[1], -1, 1)))))
    groups, cur = [], [0]
    for i in range(1, len(poses)):
        if island[i] == island[cur[0]] and np.linalg.norm(stations[i][0] - stations[cur[0]][0]) < 0.4:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    spots = []
    for g in groups:
        p = np.mean([stations[i][0] for i in g], axis=0)
        _, yaw, pitch = stations[g[len(g) // 2]]
        near = P[(np.abs(P[:, 0] - p[0]) < 1.5) & (np.abs(P[:, 2] - p[2]) < 1.5) & (P[:, 1] < p[1] - 0.3)]
        floor = float(np.percentile(near[:, 1], 8)) if len(near) > 200 else float(p[1] - 1.2)
        spots.append({"p": [round(float(x), 3) for x in p], "floor": round(floor, 3),
                      "yaw": round(yaw, 4), "pitch": round(pitch, 4), "n": len(g)})
    return spots


def write_ply(path, P, C):
    out = np.zeros(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    out["x"], out["y"], out["z"] = P[:, 0], P[:, 1], P[:, 2]
    out["red"], out["green"], out["blue"] = C[:, 0], C[:, 1], C[:, 2]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n" % len(out)).encode("ascii"))
        out.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--no-loops", action="store_true", help="skip drift correction")
    ap.add_argument("--step", type=int, default=2, help="use every Nth depth pixel (1 = densest)")
    ap.add_argument("--voxel", type=float, default=0.02, help="merge points closer than this (m)")
    ap.add_argument("--max-points", type=int, default=2_500_000)
    a = ap.parse_args()

    sess = Path(a.session)
    intr = json.loads((sess / "intrinsics.json").read_text())
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    meta = json.loads((sess / "snaps.json").read_text())
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015)
    snaps = []
    for m in meta:
        color = cv2.imread(str(sess / (m["name"] + "_color.png")))
        depth = np.load(sess / (m["name"] + "_depth_mm.npy")).astype(np.float32) / 1000.0
        snaps.append(Snap(m["name"], color, depth, sift))
    print("loaded %d snaps (%dx%d)" % (len(snaps), intr["width"], intr["height"]))
    if len(snaps) < 2:
        raise SystemExit("Need at least 2 snaps.")

    print("joining snaps:")
    poses, island, approx = build_chain(snaps, K, print)
    loops = 0 if a.no_loops else close_loops(snaps, poses, island, K, print)
    blocks = max(island) + 1

    P, C = make_cloud(snaps, poses, K, a.step, a.voxel)
    if len(P) > a.max_points:
        idx = np.random.default_rng(1).choice(len(P), a.max_points, replace=False)
        P, C = P[idx], C[idx]
    spots = make_spots(poses, island, P)
    print("%d points, %d walk-through spots, %d connected block(s), %d loop(s) closed"
          % (len(P), len(spots), blocks, loops))
    if approx:
        print("NOTE: %d snap(s) were joined only approximately (weak depth evidence); those stretches "
              "may be slightly misaligned." % approx)
    if blocks > 1:
        print("WARNING: %d snaps could not be joined to the one before. Re-shoot that stretch with a "
              "bit more overlap (smaller turns and steps) and keep the camera still." % (blocks - 1))

    fovy = math.degrees(2 * math.atan(intr["height"] / (2 * intr["fy"])))
    dot = " · "
    info = "%d snaps" % len(snaps)
    if loops:
        info += dot + "%d loop(s) closed" % loops
    if approx:
        info += dot + "%d approximate join(s)" % approx
    if blocks > 1:
        info += dot + "WARNING: %d unjoined block(s)" % (blocks - 1)
    html = (HTML.replace("__POS__", base64.b64encode(P.tobytes()).decode())
                .replace("__RGB__", base64.b64encode(C.tobytes()).decode())
                .replace("__N__", str(len(P))).replace("__FOVY__", "%.2f" % fovy)
                .replace("__SPOTS__", json.dumps(spots)).replace("__INFO__", info))
    (sess / "streetview.html").write_text(html, encoding="utf-8")
    write_ply(sess / "world.ply", P, C)
    print("wrote", sess / "streetview.html")
    print("wrote", sess / "world.ply")


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Lab Walkthrough</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:#05070a;overflow:hidden;font-family:system-ui,sans-serif;color:#ddd}
#hud{position:fixed;left:12px;top:10px;background:rgba(0,0,0,.55);padding:8px 12px;border-radius:8px;
     font-size:13px;line-height:1.55;user-select:none;max-width:330px}
button{font:inherit;color:#fff;background:#2b6cff;border:0;border-radius:6px;padding:4px 10px;margin-top:4px;cursor:pointer}
#mini{position:fixed;left:12px;bottom:12px;width:200px;height:200px;background:rgba(0,0,0,.5);
      border-radius:8px;cursor:pointer}
#msg{position:fixed;left:50%;bottom:30px;transform:translateX(-50%);background:rgba(0,0,0,.65);
     padding:6px 12px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none}
canvas{touch-action:none}
</style></head><body>
<div id="hud"><b>Lab walkthrough</b> &mdash; <span id="info"></span><br>
drag: look around &middot; scroll: zoom<br>
&uarr; / W or click a ring: move there &middot; &darr; / S: step back<br>
Tab: free-fly view of the whole map<br>
<button id="mode">Free-fly view</button>
<label style="margin-left:8px">point size <input id="ps" type="range" min="1" max="10" step="0.5" value="4"></label>
</div>
<canvas id="mini" width="400" height="400"></canvas>
<div id="msg"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
function b64(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
const N=__N__, FOVY=__FOVY__, spots=__SPOTS__;
const pos=new Float32Array(b64("__POS__").buffer), rgb=b64("__RGB__");
const col=new Float32Array(3*N); for(let i=0;i<3*N;i++) col[i]=rgb[i]/255;
document.getElementById('info').textContent='__INFO__';

const scene=new THREE.Scene(); scene.background=new THREE.Color(0x05070a);
const cam=new THREE.PerspectiveCamera(FOVY,innerWidth/innerHeight,0.03,300); cam.rotation.order='YXZ';
const ren=new THREE.WebGLRenderer({antialias:false}); ren.setSize(innerWidth,innerHeight);
ren.setPixelRatio(Math.min(devicePixelRatio,2)); document.body.appendChild(ren.domElement);
const dom=ren.domElement;
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
geo.setAttribute('color',new THREE.BufferAttribute(col,3));
const mat=new THREE.PointsMaterial({size:0.03,vertexColors:true,sizeAttenuation:true});
const cloud=new THREE.Points(geo,mat); cloud.frustumCulled=false; scene.add(cloud);
const ps=document.getElementById('ps'); ps.oninput=()=>{mat.size=ps.value*0.008;}; ps.oninput();

// ring markers on the floor at every spot
const ringGeo=new THREE.RingGeometry(0.11,0.18,32); ringGeo.rotateX(-Math.PI/2);
const rings=spots.map(s=>{
  const m=new THREE.Mesh(ringGeo,new THREE.MeshBasicMaterial({color:0x37d6ff,transparent:true,opacity:.9,depthTest:false,side:THREE.DoubleSide}));
  m.position.set(s.p[0],s.floor+0.03,s.p[2]); m.renderOrder=5; scene.add(m); return m;
});
const ringPos=spots.map(s=>new THREE.Vector3(s.p[0],s.floor+0.03,s.p[2]));

let mode='walk', cur=0, yaw=spots[0].yaw, pitch=spots[0].pitch, fov=FOVY, tween=null;
const ctl=new THREE.OrbitControls(cam,dom); ctl.enableDamping=true; ctl.enabled=false;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function applyWalk(){ cam.rotation.set(pitch,yaw,0,'YXZ'); }
function say(t){const m=document.getElementById('msg'); m.textContent=t; m.style.opacity=1; clearTimeout(say.t); say.t=setTimeout(()=>m.style.opacity=0,1600);}
function showRings(){ rings.forEach((r,i)=>r.visible=(mode==='orbit')||i!==cur); }
cam.position.set(...spots[0].p); applyWalk(); showRings();

function goTo(i){ if(i===cur||tween) return;
  tween={a:cam.position.clone(),b:new THREE.Vector3(...spots[i].p),t0:performance.now(),i}; }
function go(dir){
  const f=new THREE.Vector3(); cam.getWorldDirection(f); f.y=0; if(f.lengthSq()<1e-6) return; f.normalize().multiplyScalar(dir);
  const c=new THREE.Vector3(...spots[cur].p); let best=-1,score=1e9;
  spots.forEach((s,i)=>{ if(i===cur) return;
    const v=new THREE.Vector3(s.p[0]-c.x,0,s.p[2]-c.z), d=v.length(); if(d<0.3||d>9) return;
    const cs=v.dot(f)/d; if(cs<0.5) return; const sc=d*(2-cs); if(sc<score){score=sc;best=i;} });
  if(best<0) say(dir>0?'no spot ahead - turn towards a ring':'no spot behind'); else goTo(best);
}
function setMode(m){
  mode=m; ctl.enabled=(m==='orbit'); tween=null;
  document.getElementById('mode').textContent=(m==='walk')?'Free-fly view':'Back to walking';
  if(m==='orbit'){
    const xs=spots.map(s=>s.p[0]),ys=spots.map(s=>s.p[1]),zs=spots.map(s=>s.p[2]);
    const cx=(Math.min(...xs)+Math.max(...xs))/2, cy=(Math.min(...ys)+Math.max(...ys))/2, cz=(Math.min(...zs)+Math.max(...zs))/2;
    const ext=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...zs)-Math.min(...zs),4);
    cam.fov=FOVY; cam.updateProjectionMatrix();
    cam.position.set(cx,cy+ext*0.8,cz+ext*0.8); ctl.target.set(cx,cy,cz); ctl.update();
  } else { cam.fov=fov; cam.updateProjectionMatrix(); cam.position.set(...spots[cur].p); applyWalk(); }
  showRings();
}
document.getElementById('mode').onclick=()=>setMode(mode==='walk'?'orbit':'walk');

function screenOf(v){ const p=v.clone().project(cam); return {x:(p.x*0.5+0.5)*innerWidth,y:(1-(p.y*0.5+0.5))*innerHeight,ok:p.z<1&&p.z>-1}; }
function clickAt(x,y){
  cam.updateMatrixWorld(); let best=-1,bd=34;
  ringPos.forEach((v,i)=>{ if(i===cur&&mode==='walk') return; const s=screenOf(v); if(!s.ok) return;
    const d=Math.hypot(s.x-x,s.y-y); if(d<bd){bd=d;best=i;} });
  if(best>=0){ if(mode==='orbit'){ cur=best; setMode('walk'); } else goTo(best); }
}
let drag=null;
dom.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,sx:e.clientX,sy:e.clientY};});
dom.addEventListener('pointermove',e=>{
  if(!drag||mode!=='walk') return; const k=0.0035*(cam.fov/60);
  yaw+=(e.clientX-drag.x)*k; pitch=clamp(pitch+(e.clientY-drag.y)*k,-1.4,1.4);
  drag.x=e.clientX; drag.y=e.clientY; applyWalk(); });
addEventListener('pointerup',e=>{ if(!drag) return;
  const moved=Math.hypot(e.clientX-drag.sx,e.clientY-drag.sy); drag=null; if(moved<5) clickAt(e.clientX,e.clientY); });
dom.addEventListener('wheel',e=>{ if(mode!=='walk') return; e.preventDefault();
  fov=clamp(fov+e.deltaY*0.04,25,100); cam.fov=fov; cam.updateProjectionMatrix(); },{passive:false});
addEventListener('keydown',e=>{
  if(e.key==='Tab'){ e.preventDefault(); setMode(mode==='walk'?'orbit':'walk'); return; }
  if(mode!=='walk') return;
  if(e.key==='ArrowUp'||e.key==='w'||e.key==='W') go(1);
  else if(e.key==='ArrowDown'||e.key==='s'||e.key==='S') go(-1);
  else if(e.key==='ArrowLeft'||e.key==='a'||e.key==='A'){ yaw+=0.15; applyWalk(); }
  else if(e.key==='ArrowRight'||e.key==='d'||e.key==='D'){ yaw-=0.15; applyWalk(); } });
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);});

// mini-map
const mini=document.getElementById('mini'),mx=mini.getContext('2d');
const xs=spots.map(s=>s.p[0]),zs=spots.map(s=>s.p[2]);
const minX=Math.min(...xs)-1,maxX=Math.max(...xs)+1,minZ=Math.min(...zs)-1,maxZ=Math.max(...zs)+1;
const sc=Math.min(360/(maxX-minX),360/(maxZ-minZ));
const mm=p=>[20+(p[0]-minX)*sc,20+(p[2]-minZ)*sc];
function drawMini(){
  mx.clearRect(0,0,400,400); mx.strokeStyle='rgba(255,255,255,.35)'; mx.lineWidth=2; mx.beginPath();
  spots.forEach((s,i)=>{const [x,y]=mm(s.p); i?mx.lineTo(x,y):mx.moveTo(x,y);}); mx.stroke();
  spots.forEach((s,i)=>{const [x,y]=mm(s.p); mx.fillStyle=(i===cur)?'#ffb000':'#37d6ff'; mx.beginPath(); mx.arc(x,y,(i===cur)?9:6,0,6.283); mx.fill();});
  const [x,y]=mm(cam.position.toArray());
  if(mode==='walk'){ mx.strokeStyle='#ffb000'; mx.lineWidth=4; mx.beginPath(); mx.moveTo(x,y);
    mx.lineTo(x-Math.sin(yaw)*30,y-Math.cos(yaw)*30); mx.stroke(); }
}
mini.addEventListener('click',e=>{ const r=mini.getBoundingClientRect(); const px=(e.clientX-r.left)/r.width*400, py=(e.clientY-r.top)/r.height*400;
  let best=-1,bd=30; spots.forEach((s,i)=>{const [x,y]=mm(s.p); const d=Math.hypot(x-px,y-py); if(d<bd){bd=d;best=i;}});
  if(best>=0){ if(mode==='orbit'){cur=best;setMode('walk');} else goTo(best); } });

(function loop(){
  requestAnimationFrame(loop);
  if(tween){ const u=Math.min(1,(performance.now()-tween.t0)/700), e=u<.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
    cam.position.lerpVectors(tween.a,tween.b,e); if(u>=1){cur=tween.i;tween=null;showRings();} }
  if(mode==='orbit') ctl.update();
  ren.render(scene,cam); drawMini();
})();
</script></body></html>
"""


if __name__ == "__main__":
    main()
