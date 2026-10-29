"""Turn one capture (colour + depth in mm + intrinsics) into a smooth depth map you can measure on.

Usage:  python make_measure_view.py path\\to\\shots\\drone      (prefix: expects drone_color.png,
        drone_depth_mm.npy, drone_intrinsics.json)
Writes  <prefix>_measure.html : open it, click two points on either picture to get the distance.
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

from make_depth_map import fill_holes, shifted

OFFS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def smooth_edge_aware(depth, iters=4, tol=0.03):
    """Average each pixel with neighbours at a similar depth (within tol), so flat surfaces
    become smooth while object edges stay sharp."""
    d = depth.copy()
    with np.errstate(invalid="ignore"):
        for _ in range(iters):
            valid = np.isfinite(d)
            num = np.where(valid, d, 0.0).astype(np.float64)
            cnt = valid.astype(np.float64)
            for dy, dx in OFFS:
                n = shifted(d, dy, dx, np.inf)
                ok = np.isfinite(n) & valid & (np.abs(n - d) <= tol * d)
                num += np.where(ok, n, 0.0)
                cnt += ok
            d = np.where(valid, num / np.maximum(cnt, 1), np.inf)
    return d


def main():
    prefix = Path(sys.argv[1])
    color_png = prefix.with_name(prefix.name + "_color.png")
    raw = np.load(prefix.with_name(prefix.name + "_depth_mm.npy")).astype(np.float32)
    k = json.loads(prefix.with_name(prefix.name + "_intrinsics.json").read_text())
    h, w = raw.shape
    print(f"depth {w}x{h}, valid {100.0 * (raw > 0).mean():.0f}%")

    measured = raw > 0
    d = np.where(measured, raw, np.inf).astype(np.float32)
    d = fill_holes(d, 12)
    d = smooth_edge_aware(d, 4)
    ok = np.isfinite(d)
    out_mm = np.where(ok, np.round(d), 0).astype(np.uint16)
    lo, hi = np.percentile(d[ok], [1, 99]) if ok.any() else (300, 4000)

    html = (HTML.replace("__DEPTH__", base64.b64encode(out_mm.tobytes()).decode())
                .replace("__MASK__", base64.b64encode(measured.astype(np.uint8).tobytes()).decode())
                .replace("__COLOR__", base64.b64encode(color_png.read_bytes()).decode())
                .replace("__W__", str(w)).replace("__H__", str(h))
                .replace("__FX__", str(k["fx"])).replace("__FY__", str(k["fy"]))
                .replace("__CX__", str(k["cx"])).replace("__CY__", str(k["cy"]))
                .replace("__LO__", f"{lo:.1f}").replace("__HI__", f"{hi:.1f}"))
    out = prefix.with_name(prefix.name + "_measure.html")
    out.write_text(html, encoding="utf-8")
    print("wrote", out)


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Depth Measure</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;background:#000;color:#ddd;font-family:system-ui,sans-serif}
#bar{padding:8px 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-size:14px}
button{font:inherit;color:#fff;background:#2b6cff;border:0;border-radius:6px;padding:5px 12px;cursor:pointer}
button.warn{background:#8a2b2b}
#panes{display:flex;gap:6px;padding:0 6px}
.pane{flex:1;min-width:0}
canvas{width:100%;display:block;cursor:crosshair;background:#111}
.cap{font-size:12px;color:#999;padding:2px 0}
#list{padding:8px 12px;font-variant-numeric:tabular-nums;font-size:14px}
#list div{border-top:1px solid #333;padding:3px 0}
</style></head><body>
<div id="bar"><b>Click two points to measure</b>
<button id="undo">Undo</button><button id="clear" class="warn">Clear</button>
<span id="hover" style="color:#ffb000">hover over the picture</span></div>
<div id="panes">
 <div class="pane"><div class="cap">camera</div><canvas id="c1"></canvas></div>
 <div class="pane"><div class="cap">depth (bright = near)</div><canvas id="c2"></canvas></div>
</div>
<div id="list"></div>
<script>
function b64(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a;}
const W=__W__,H=__H__,FX=__FX__,FY=__FY__,CX=__CX__,CY=__CY__,LO=__LO__,HI=__HI__;
const depth=new Uint16Array(b64("__DEPTH__").buffer), mask=b64("__MASK__");
const c1=document.getElementById('c1'),c2=document.getElementById('c2');
for(const c of [c1,c2]){c.width=W;c.height=H;}
const x1=c1.getContext('2d'),x2=c2.getContext('2d');
// depth picture
const dimg=x2.createImageData(W,H);
for(let i=0;i<W*H;i++){
  const d=depth[i]; let g=0;
  if(d>0){const t=Math.min(1,Math.max(0,(d-LO)/(HI-LO)));g=255-230*t;}
  dimg.data[4*i]=dimg.data[4*i+1]=dimg.data[4*i+2]=g; dimg.data[4*i+3]=255;
}
const colorImg=new Image(); colorImg.src="data:image/png;base64,__COLOR__";
let pts=[], pairs=[], hov=null;

// Median depth (mm) of a small window; returns {z, estimated}.
function sample(u,v){
  const vals=[]; let meas=0;
  for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
    const x=u+dx,y=v+dy; if(x<0||y<0||x>=W||y>=H) continue;
    const i=y*W+x; if(depth[i]>0){vals.push(depth[i]); if(mask[i]) meas++;}
  }
  if(!vals.length) return null;
  vals.sort((a,b)=>a-b);
  return {z:vals[vals.length>>1], estimated: meas<5};
}
function to3d(u,v,z){ z=z/1000; return [(u-CX)*z/FX,(v-CY)*z/FY,z]; }
function fmt(m){ return m>=1 ? m.toFixed(2)+' m' : (m*100).toFixed(1)+' cm'; }
function pixel(e,c){
  const r=c.getBoundingClientRect();
  return [Math.floor((e.clientX-r.left)/r.width*W), Math.floor((e.clientY-r.top)/r.height*H)];
}
function marker(ctx,p,col){
  ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.u,p.v,7,0,6.283);ctx.stroke();
  ctx.beginPath();ctx.moveTo(p.u-12,p.v);ctx.lineTo(p.u+12,p.v);ctx.moveTo(p.u,p.v-12);ctx.lineTo(p.u,p.v+12);ctx.stroke();
}
function draw(){
  x1.drawImage(colorImg,0,0,W,H); x2.putImageData(dimg,0,0);
  for(const ctx of [x1,x2]){
    for(const pr of pairs){
      ctx.strokeStyle='#ffb000';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(pr.a.u,pr.a.v);ctx.lineTo(pr.b.u,pr.b.v);ctx.stroke();
      marker(ctx,pr.a,'#ffb000');marker(ctx,pr.b,'#ffb000');
      const mx=(pr.a.u+pr.b.u)/2,my=(pr.a.v+pr.b.v)/2-10;
      ctx.font='bold 18px sans-serif';ctx.fillStyle='#000';ctx.fillRect(mx-4,my-17,ctx.measureText(pr.txt).width+8,24);
      ctx.fillStyle='#ffb000';ctx.fillText(pr.txt,mx,my);
    }
    for(const p of pts) marker(ctx,p,'#00e0ff');
    if(hov) marker(ctx,hov,'rgba(255,255,255,.7)');
  }
}
function makePoint(u,v){
  const s=sample(u,v); if(!s) return null;
  return {u,v,z:s.z,est:s.estimated,xyz:to3d(u,v,s.z)};
}
function refresh(){
  document.getElementById('list').innerHTML=pairs.map((p,i)=>{
    const dx=p.b.xyz[0]-p.a.xyz[0],dy=p.b.xyz[1]-p.a.xyz[1],dz=p.b.xyz[2]-p.a.xyz[2];
    const flag=(p.a.est||p.b.est)?' <span style="color:#ff6a6a">(a point is in a gap: estimated)</span>':'';
    return `<div><b>#${i+1}: ${fmt(Math.hypot(dx,dy,dz))}</b> &middot; across ${fmt(Math.hypot(dx,dy))}
      &middot; depth difference ${fmt(Math.abs(dz))} &middot; A at ${fmt(p.a.xyz[2])}, B at ${fmt(p.b.xyz[2])}${flag}</div>`;
  }).join('');
}
function onClick(e,c){
  const [u,v]=pixel(e,c); const p=makePoint(u,v); if(!p) return;
  if(pts.length===0){ pts.push(p); }
  else{
    const a=pts.pop(); const dx=p.xyz[0]-a.xyz[0],dy=p.xyz[1]-a.xyz[1],dz=p.xyz[2]-a.xyz[2];
    pairs.push({a,b:p,txt:fmt(Math.hypot(dx,dy,dz))}); refresh();
  }
  draw();
}
function onMove(e,c){
  const [u,v]=pixel(e,c); const s=sample(u,v); hov={u,v};
  document.getElementById('hover').textContent = s
    ? `pixel (${u},${v})  distance from camera: ${fmt(s.z/1000)}${s.estimated?' (estimated, gap)':''}`
    : `pixel (${u},${v})  no depth here`;
  draw();
}
for(const c of [c1,c2]){
  c.addEventListener('click',e=>onClick(e,c));
  c.addEventListener('mousemove',e=>onMove(e,c));
  c.addEventListener('mouseleave',()=>{hov=null;draw();});
}
document.getElementById('undo').onclick=()=>{ if(pts.length) pts.pop(); else pairs.pop(); refresh(); draw(); };
document.getElementById('clear').onclick=()=>{ pts=[];pairs=[];refresh();draw(); };
colorImg.onload=draw; draw();
</script></body></html>
"""


if __name__ == "__main__":
    main()
