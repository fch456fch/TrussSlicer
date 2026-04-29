"""3D 路径数据结构与可视化。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from shapely.geometry import MultiPolygon, Polygon


KIND_COLORS = {
    "bottom_solid": "#888888",
    "top_solid": "#888888",
    "sparse_infill": "#ff7f0e",
    "truss_anchor": "#1f77b4",
    "truss_body_low": "#2ca02c",
    "truss_body_high": "#d62728",
    "wall": "#000000",
    "contour": "#aaaaaa",
}

MARKER_KINDS = {"bottom_solid", "top_solid", "sparse_infill", "truss_anchor",
                "truss_body_low", "truss_body_high"}


@dataclass
class LayerPaths:
    """一层的所有打印路径。每个 path 是 (N, 3) 的 numpy 数组。"""
    index: int
    z: float
    kind: str = "contour"
    paths: list[np.ndarray] = field(default_factory=list)


def polygon_to_paths(poly: Polygon, z: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    rings = [poly.exterior] + list(poly.interiors)
    for ring in rings:
        xy = np.asarray(ring.coords)
        if len(xy) < 2:
            continue
        zs = np.full((len(xy), 1), z)
        out.append(np.hstack([xy, zs]))
    return out


def multipolygon_to_layer_paths(
    mp: MultiPolygon, z: float, index: int, kind: str = "contour"
) -> LayerPaths:
    paths: list[np.ndarray] = []
    for poly in mp.geoms:
        paths.extend(polygon_to_paths(poly, z))
    return LayerPaths(index=index, z=z, kind=kind, paths=paths)


def _flatten_segments(layers: list[LayerPaths]) -> list[dict]:
    segments = []
    for lp in layers:
        for path in lp.paths:
            coords = path.tolist() if isinstance(path, np.ndarray) else path
            segments.append({
                "layer": lp.index,
                "z": round(lp.z, 4),
                "kind": lp.kind,
                "coords": [[round(c, 4) for c in pt] for pt in coords],
            })
    return segments


def render_html(layers: list[LayerPaths], out_path: str, title: str = "truss-slicer") -> None:
    segments = _flatten_segments(layers)
    if not segments:
        return

    layer_indices = sorted(set(s["layer"] for s in segments))
    max_layer = max(layer_indices) if layer_indices else 0

    colors_json = json.dumps(KIND_COLORS)
    marker_kinds_json = json.dumps(list(MARKER_KINDS))
    segments_json = json.dumps(segments)

    html = _INTERACTIVE_HTML.replace("__SEGMENTS__", segments_json)
    html = html.replace("__COLORS__", colors_json)
    html = html.replace("__MARKER_KINDS__", marker_kinds_json)
    html = html.replace("__MAX_LAYER__", str(max_layer))
    html = html.replace("__TITLE__", title)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


_INTERACTIVE_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{margin:0;font-family:system-ui,sans-serif;background:#f5f5f5;color:#333;overflow:hidden}
#plot{position:absolute;top:0;left:0;right:60px;bottom:50px}
#layerPanel{position:fixed;top:0;right:0;bottom:50px;width:60px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  background:#fff;border-left:1px solid #ddd;padding:8px 0;z-index:100}
#layerPanel label{font-size:11px;margin-bottom:4px;white-space:nowrap}
#layerSlider{writing-mode:vertical-lr;direction:rtl;height:calc(100% - 40px);
  accent-color:#e94560;cursor:pointer}
#stepPanel{position:fixed;bottom:0;left:0;right:60px;height:50px;
  display:flex;align-items:center;gap:12px;padding:0 16px;
  background:#fff;border-top:1px solid #ddd;z-index:100}
#stepPanel label{font-size:11px;min-width:100px;white-space:nowrap}
#stepSlider{flex:1;accent-color:#e94560;cursor:pointer}
</style></head><body>
<div id="plot"></div>
<div id="layerPanel">
  <label>L <span id="layerVal">0</span></label>
  <input type="range" id="layerSlider" min="0" max="__MAX_LAYER__" value="__MAX_LAYER__" step="1">
</div>
<div id="stepPanel">
  <label>Step <span id="stepVal">0</span>/<span id="stepMax">0</span></label>
  <input type="range" id="stepSlider" min="0" max="0" value="0" step="1">
</div>
<script>
const ALL=__SEGMENTS__;
const COLORS=__COLORS__;
const MK=new Set(__MARKER_KINDS__);
const layerSlider=document.getElementById('layerSlider');
const stepSlider=document.getElementById('stepSlider');
const layerVal=document.getElementById('layerVal');
const stepVal=document.getElementById('stepVal');
const stepMax=document.getElementById('stepMax');

let currentLayout={title:'__TITLE__',
  scene:{xaxis:{title:'X'},yaxis:{title:'Y'},zaxis:{title:'Z'},aspectmode:'data'},
  margin:{l:0,r:0,t:30,b:0},
  paper_bgcolor:'#f5f5f5',
  font:{color:'#333'},
  showlegend:true};

Plotly.newPlot('plot',[],currentLayout,{responsive:true});

function segsByLayer(){
  const m={};
  ALL.forEach((s,i)=>{s._gi=i;if(!m[s.layer])m[s.layer]=[];m[s.layer].push(s)});
  return m;
}
const byLayer=segsByLayer();
const layerKeys=Object.keys(byLayer).map(Number).sort((a,b)=>a-b);

function update(){
  const curL=+layerSlider.value;
  layerVal.textContent=curL;
  const below=[];
  const curSegs=byLayer[curL]||[];
  layerKeys.forEach(l=>{if(l<curL && byLayer[l])below.push(...byLayer[l])});
  const totalSteps=curSegs.length;
  stepSlider.max=totalSteps;
  if(+stepSlider.value>totalSteps)stepSlider.value=totalSteps;
  stepMax.textContent=totalSteps;
  const nShow=+stepSlider.value;
  stepVal.textContent=nShow;
  const curShow=curSegs.slice(0,nShow);
  const traces=[];
  // previous layers — faded
  const belowKinds={};
  below.forEach(s=>{
    if(!belowKinds[s.kind])belowKinds[s.kind]={x:[],y:[],z:[]};
    const g=belowKinds[s.kind];
    s.coords.forEach(c=>{g.x.push(c[0]);g.y.push(c[1]);g.z.push(c[2])});
    g.x.push(null);g.y.push(null);g.z.push(null);
  });
  Object.keys(belowKinds).forEach(k=>{
    const g=belowKinds[k];
    const c=COLORS[k]||'#666';
    const r=parseInt(c.slice(1,3),16),gr=parseInt(c.slice(3,5),16),b=parseInt(c.slice(5,7),16);
    traces.push({type:'scatter3d',mode:'lines',name:k+' (prev)',
      x:g.x,y:g.y,z:g.z,
      line:{color:'rgba('+r+','+gr+','+b+',0.15)',width:1},
      hoverinfo:'skip',showlegend:false});
  });
  // current layer — full opacity
  const curKinds={};
  curShow.forEach(s=>{
    if(!curKinds[s.kind])curKinds[s.kind]={x:[],y:[],z:[]};
    const g=curKinds[s.kind];
    s.coords.forEach(c=>{g.x.push(c[0]);g.y.push(c[1]);g.z.push(c[2])});
    g.x.push(null);g.y.push(null);g.z.push(null);
  });
  Object.keys(curKinds).forEach(k=>{
    const g=curKinds[k];
    traces.push({type:'scatter3d',mode:'lines',name:k,
      x:g.x,y:g.y,z:g.z,
      line:{color:COLORS[k]||'#666',width:2},
      hoverinfo:'name'});
  });
  if(nShow>0){
    const last=curSegs[nShow-1];
    const c=last.coords;
    if(MK.has(last.kind)&&c.length>=2){
      traces.push({type:'scatter3d',mode:'markers',name:'start',
        x:[c[0][0]],y:[c[0][1]],z:[c[0][2]],
        marker:{color:'#00ff88',size:5,symbol:'circle'},showlegend:false});
      const e=c[c.length-1];
      traces.push({type:'scatter3d',mode:'markers',name:'end',
        x:[e[0]],y:[e[1]],z:[e[2]],
        marker:{color:'#ff4466',size:5,symbol:'circle'},showlegend:false});
    }
  }
  Plotly.react('plot',traces,currentLayout);
}

layerSlider.addEventListener('input',()=>{stepSlider.value=stepSlider.max;update()});
stepSlider.addEventListener('input',()=>{update()});

document.addEventListener('keydown',e=>{
  if(e.key==='ArrowUp'){e.preventDefault();layerSlider.value=Math.min(+layerSlider.value+1,+layerSlider.max);stepSlider.value=stepSlider.max;update()}
  if(e.key==='ArrowDown'){e.preventDefault();layerSlider.value=Math.max(+layerSlider.value-1,0);stepSlider.value=stepSlider.max;update()}
  if(e.key==='ArrowRight'){e.preventDefault();stepSlider.value=Math.min(+stepSlider.value+1,+stepSlider.max);update()}
  if(e.key==='ArrowLeft'){e.preventDefault();stepSlider.value=Math.max(+stepSlider.value-1,0);update()}
});

layerSlider.value=layerSlider.max;
update();
</script></body></html>"""