"""Turn an RTAB-Map point cloud (.ply) into a solid, straightened, coloured 3D model of the room.

Usage:  python make_lab_model.py path\\to\\map_xxx_cloud.ply [--keep-ceiling] [--detail 9]

What it does
  1. Finds the true "up" (RTAB-Map's map is tilted by however the camera was held at the start)
     and turns the map so the floor/ceiling are level and the walls line up with the axes.
  2. Cuts the room off just above where the camera started (--cut, default 0.2 m) so you can look
     down into it; --keep-ceiling keeps everything.
  3. Cleans stray points and builds a continuous surface (Poisson reconstruction), trimming
     surface that has no real data behind it, keeping the colours of the photos.
Writes  <name>_model.html  (orbit viewer)  and  <name>_model.ply  (open in MeshLab / Blender).
"""
import argparse
import base64
import math
from pathlib import Path

import numpy as np
import open3d as o3d

from make_bw_world import read_ply


def find_up(xyz, n, rng):
    """Vertical axis: the direction most surface normals agree on, refined on the densest flat layer."""
    sub = rng.choice(len(n), min(600_000, len(n)), replace=False)
    _, V = np.linalg.eigh((n[sub].T @ n[sub]) / len(sub))
    ax = V[:, 2]
    X, N = xyz[sub], n[sub]
    for _ in range(4):
        h = X @ ax
        flat = np.abs(N @ ax) > 0.9
        hist, edges = np.histogram(h[flat], bins=160)
        k = int(np.argmax(hist))
        centre = (edges[k] + edges[k + 1]) / 2
        pts = X[flat & (np.abs(h - centre) < 0.12)]
        if len(pts) < 500:
            break
        c = pts.mean(0)
        nn = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
        ax = nn if nn @ ax > 0 else -nn
    # the camera looks mostly at the ceiling: the side with more flat surface is "up"
    h = xyz[sub] @ ax
    flat = np.abs(N @ ax) > 0.9
    if (flat & (h > 0.3)).sum() < (flat & (h < -0.3)).sum():
        ax = -ax
    return ax / np.linalg.norm(ax)


def level_rotation(xyz, n, up):
    """Rotation taking `up` to +y and the room's dominant wall direction to +x."""
    e1 = np.cross(up, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 0.2:
        e1 = np.cross(up, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    walls = np.abs(n @ up) < 0.25
    phi = np.arctan2(n[walls] @ e2, n[walls] @ e1)
    theta = np.angle(np.exp(4j * phi).mean()) / 4 if walls.sum() > 1000 else 0.0
    x = math.cos(theta) * e1 + math.sin(theta) * e2
    z = np.cross(x, up)
    return np.stack([x, up, z])


def layers(h, counts_min=0.25, bins=200):
    """Heights of dense flat layers (floors, ceilings, table tops), lowest to highest."""
    hist, edges = np.histogram(h, bins=bins)
    peak = hist.max()
    out = []
    for i in range(1, bins - 1):
        if hist[i] >= counts_min * peak and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            out.append((float((edges[i] + edges[i + 1]) / 2), int(hist[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("--keep-ceiling", action="store_true")
    ap.add_argument("--cut", type=float, default=0.2,
                    help="cut-away height in metres above where the camera started (default 0.2)")
    ap.add_argument("--detail", type=int, default=9, help="Poisson depth: 8 = smooth, 10 = detailed (slow)")
    ap.add_argument("--voxel", type=float, default=0.03, help="point spacing used for the surface (m)")
    ap.add_argument("--trim", type=float, default=0.12, help="drop this fraction of the least-supported surface")
    ap.add_argument("--triangles", type=int, default=700_000, help="triangles kept in the web page")
    a = ap.parse_args()

    src = Path(a.ply)
    rng = np.random.default_rng(0)
    d = read_ply(src)
    names = d.dtype.names
    xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    n = np.stack([d["nx"], d["ny"], d["nz"]], 1).astype(np.float64)
    rgb = np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.uint8)
    print("read %d points" % len(xyz))

    up = find_up(xyz, n, rng)
    R = level_rotation(xyz, n, up)
    P, N = xyz @ R.T, n @ R.T
    tilt = math.degrees(math.acos(np.clip(up @ np.array([0, 0, 1.0]), -1, 1)))
    print("straightened the map (it was tilted %.0f deg)" % tilt)

    flat = np.abs(N[:, 1]) > 0.9
    lay = layers(P[flat, 1])
    floor_y = lay[0][0] if lay else float(np.percentile(P[:, 1], 2))
    ceil_y = lay[-1][0] if len(lay) > 1 else float(np.percentile(P[:, 1], 98))
    print("flat layers at heights (m): %s" % ", ".join("%.2f" % l[0] for l in lay))
    print("floor ~ %.2f m, ceiling ~ %.2f m (room height about %.1f m)" % (floor_y, ceil_y, ceil_y - floor_y))

    keep = np.ones(len(P), bool)
    keep &= (P[:, 1] > np.percentile(P[:, 1], 0.5) - 0.2)                 # stray points far below
    if not a.keep_ceiling:
        # Cut-away view: the ceiling is uneven in the map (drift), so cut at one height instead,
        # a little above where the camera started. Desks, monitors, chairs and wall bases stay.
        keep &= (P[:, 1] < a.cut)
    else:
        keep &= (P[:, 1] < ceil_y + 0.3)
    P, N, C = P[keep], N[keep], rgb[keep]
    print("%d points after removing the ceiling and anything outside the room" % len(P))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.colors = o3d.utility.Vector3dVector(C / 255.0)
    pcd.normals = o3d.utility.Vector3dVector(N)
    pcd = pcd.voxel_down_sample(a.voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.2)
    print("%d points used for the surface" % len(pcd.points))

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=a.detail)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, a.trim))
    box = pcd.get_axis_aligned_bounding_box()
    box = o3d.geometry.AxisAlignedBoundingBox(box.min_bound - 0.1, box.max_bound + 0.1)
    mesh = mesh.crop(box)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print("surface: %d triangles" % len(mesh.triangles))

    out_ply = src.with_name(src.stem.replace("_cloud", "") + "_model.ply")
    o3d.io.write_triangle_mesh(str(out_ply), mesh)
    print("wrote", out_ply)

    web = mesh
    if len(mesh.triangles) > a.triangles:
        web = mesh.simplify_quadric_decimation(a.triangles)
    V = np.asarray(web.vertices, np.float32)
    F = np.asarray(web.triangles, np.uint32)
    Cc = (np.clip(np.asarray(web.vertex_colors), 0, 1) * 255).astype(np.uint8)
    centre = (V.min(0) + V.max(0)) / 2
    V = V - centre
    extent = float(np.linalg.norm(V.max(0) - V.min(0)))
    html = (HTML.replace("__V__", base64.b64encode(V.tobytes()).decode())
                .replace("__C__", base64.b64encode(np.ascontiguousarray(Cc).tobytes()).decode())
                .replace("__F__", base64.b64encode(F.tobytes()).decode())
                .replace("__NV__", str(len(V))).replace("__NF__", str(len(F)))
                .replace("__EXT__", "%.2f" % extent)
                .replace("__DIM__", "%.1f x %.1f x %.1f m" % tuple(V.max(0) - V.min(0))))
    out_html = src.with_name(src.stem.replace("_cloud", "") + "_model.html")
    out_html.write_text(html, encoding="utf-8")
    print("wrote", out_html, "(%d triangles in the page)" % len(F))


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Lab 3D Model</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;height:100%;background:#0b0d10;overflow:hidden;font-family:system-ui,sans-serif;color:#ddd}
#hud{position:fixed;left:12px;top:10px;background:rgba(0,0,0,.55);padding:8px 12px;border-radius:8px;font-size:13px;line-height:1.6;user-select:none}
button{font:inherit;color:#fff;background:#2b6cff;border:0;border-radius:6px;padding:4px 10px;margin:4px 4px 0 0;cursor:pointer}
button.off{background:#444}
</style></head><body>
<div id="hud"><b>Lab 3D model</b> &mdash; __DIM__<br>
drag: orbit &middot; wheel: zoom &middot; right-drag: pan<br>
<button id="b_photo">Photo colours</button><button id="b_shade" class="off">Shaded</button><button id="b_wire" class="off">Wireframe</button><br>
<label>brightness <input id="br" type="range" min="0.6" max="2" step="0.05" value="1.2"></label></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
function b64(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
const NV=__NV__,NF=__NF__,EXT=__EXT__;
const pos=new Float32Array(b64("__V__").buffer), idx=new Uint32Array(b64("__F__").buffer), col=b64("__C__");
const scene=new THREE.Scene(); scene.background=new THREE.Color(0x0b0d10);
const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.05,EXT*20);
cam.position.set(EXT*0.45,EXT*0.5,EXT*0.55);
const ren=new THREE.WebGLRenderer({antialias:true}); ren.setSize(innerWidth,innerHeight);
ren.setPixelRatio(Math.min(devicePixelRatio,2)); document.body.appendChild(ren.domElement);
const ctl=new THREE.OrbitControls(cam,ren.domElement); ctl.enableDamping=true;
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
geo.setAttribute('color',new THREE.BufferAttribute(col,3,true));
geo.setIndex(new THREE.BufferAttribute(idx,1));
geo.computeVertexNormals();
const mats={
  photo:new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide}),
  shade:new THREE.MeshStandardMaterial({vertexColors:true,side:THREE.DoubleSide,roughness:.9,metalness:0}),
  wire:new THREE.MeshBasicMaterial({vertexColors:true,wireframe:true})
};
const mesh=new THREE.Mesh(geo,mats.photo); scene.add(mesh);
scene.add(new THREE.HemisphereLight(0xffffff,0x444455,0.9));
const head=new THREE.DirectionalLight(0xffffff,0.8); scene.add(head);
function use(k){ mesh.material=mats[k]; for(const [id,key] of [['b_photo','photo'],['b_shade','shade'],['b_wire','wire']]) document.getElementById(id).classList.toggle('off',key!==k); }
document.getElementById('b_photo').onclick=()=>use('photo');
document.getElementById('b_shade').onclick=()=>use('shade');
document.getElementById('b_wire').onclick=()=>use('wire');
const br=document.getElementById('br'); br.oninput=()=>{ren.toneMappingExposure=parseFloat(br.value); const v=parseFloat(br.value); mats.photo.color.setScalar(v);}; br.oninput();
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight);});
(function loop(){requestAnimationFrame(loop);head.position.copy(cam.position);ctl.update();ren.render(scene,cam);})();
</script></body></html>
"""


if __name__ == "__main__":
    main()
