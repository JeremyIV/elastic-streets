"""Pull a 24-hour sparse anchor-grid drive-time dataset from Mapbox
depart_at typical traffic. Freeway anchors chained along motorways carry
the fast skeleton; Delaunay fabric covers the surface grid; direct
long-range pairs each hour calibrate the per-leg endpoint offset.

Usage: anchor_pull.py [graphml] [out_json] [K_surface]
Writes anchors/edges/per-hour times; resumable (skips keys already saved).
"""

import json
import pathlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial import Delaunay

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "la_full.graphml")
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data" / "la_pull.json")
K_SURF = int(sys.argv[3]) if len(sys.argv) > 3 else 160
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
EDGE_MAX_M = 4500
FWY_CHAIN_M = 5500
FWY_SPACING_M = 3000
N_VAL = 10
DAY = "2026-06-17"   # a typical Wednesday

print("loading graph...")
G = ox.load_graphml(GRAPH)
Gp = ox.project_graph(G)

FWY = {"motorway"}
def edge_hwys(d):
    hw = d.get("highway")
    return hw if isinstance(hw, list) else [hw]

art_nodes = sorted(G.nodes)
Gf = nx.Graph()
for u, v, d in Gp.edges(data=True):
    if any(h in FWY for h in edge_hwys(d)):
        w = float(d["length"])
        if not Gf.has_edge(u, v) or Gf[u][v]["w"] > w:
            Gf.add_edge(u, v, w=w)
fn = sorted(Gf.nodes)
FXY = np.array([[Gp.nodes[n]["x"], Gp.nodes[n]["y"]] for n in fn])
fsel = [0]
fd = np.linalg.norm(FXY - FXY[0], axis=1)
while fd.max() > FWY_SPACING_M:
    j = int(fd.argmax())
    fsel.append(j)
    fd = np.minimum(fd, np.linalg.norm(FXY - FXY[j], axis=1))
fanchors = [fn[i] for i in fsel]
print(f"{len(fanchors)} freeway anchors")

fwy_edges = set()
for a in range(len(fanchors)):
    dists = nx.single_source_dijkstra_path_length(
        Gf, fanchors[a], cutoff=FWY_CHAIN_M, weight="w")
    for b in range(len(fanchors)):
        if a < b and fanchors[b] in dists:
            fwy_edges.add((a, b))

XY = np.array([[Gp.nodes[n]["x"], Gp.nodes[n]["y"]] for n in art_nodes])
sel_nodes = list(fanchors)
axy_list = [[Gp.nodes[n]["x"], Gp.nodes[n]["y"]] for n in fanchors]
dmin = np.min(np.linalg.norm(
    XY[:, None, :] - np.array(axy_list)[None, :, :], axis=2), axis=1)
while len(sel_nodes) < K_SURF + len(fanchors):
    j = int(dmin.argmax())
    sel_nodes.append(art_nodes[j])
    axy_list.append([XY[j][0], XY[j][1]])
    dmin = np.minimum(dmin, np.linalg.norm(XY - XY[j], axis=1))
axy = np.array(axy_list)
all_ = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in sel_nodes])
K = len(sel_nodes)
print(f"{K} anchors, min spacing {dmin.max():.0f} m floor")

tri = Delaunay(axy)
edges = set(fwy_edges)
for s in tri.simplices:
    for a in range(3):
        i, j = sorted((int(s[a]), int(s[(a + 1) % 3])))
        if np.linalg.norm(axy[i] - axy[j]) < EDGE_MAX_M:
            edges.add((i, j))
edges = sorted(edges)
print(f"{len(edges)} edges; ~{24 * (2 * len(edges) + N_VAL)} requests")

rng = np.random.default_rng(7)
cand = [(i, j) for i in range(K) for j in range(i + 1, K)
        if np.linalg.norm(axy[i] - axy[j]) > 25000]
val_pairs = [tuple(int(x) for x in cand[i])
             for i in rng.choice(len(cand), N_VAL, replace=False)]

# resume support: keep whatever a previous run already fetched
state = {"anchor_nodes": [str(n) for n in sel_nodes],
         "anchors_ll": all_.tolist(), "anchors_xy": axy.tolist(),
         "edges": [list(e) for e in edges],
         "n_fwy_anchors": len(fanchors),
         "val_pairs": [list(p) for p in val_pairs],
         "day": DAY, "times": {f"{h:02d}": {} for h in range(24)}}
if OUT.exists():
    old = json.load(open(OUT))
    if old.get("edges") == state["edges"]:
        state["times"] = old["times"]
        print("resuming: kept", sum(len(v) for v in state["times"].values()),
              "saved results")

def mapbox(i, j, t):
    time.sleep(1.0)
    url = (f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
           f"{all_[i][0]:.6f},{all_[i][1]:.6f};{all_[j][0]:.6f},{all_[j][1]:.6f}"
           f"?access_token={TOKEN}&depart_at={t}&overview=false")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                d = json.load(r)
            return d["routes"][0]["duration"] if d.get("routes") else None
        except Exception:
            if attempt == 3:
                return None
            time.sleep(3 * (attempt + 1))

jobs = []
for h in range(24):
    hh, t = f"{h:02d}", f"{DAY}T{h:02d}:00"
    have = state["times"][hh]
    for (i, j) in edges:
        for (a, b) in ((i, j), (j, i)):
            if f"{a}-{b}" not in have:
                jobs.append((hh, f"{a}-{b}", a, b, t))
    for (i, j) in val_pairs:
        if f"V{i}-{j}" not in have:
            jobs.append((hh, f"V{i}-{j}", i, j, t))
print(f"{len(jobs)} requests to run")

done = 0
with ThreadPoolExecutor(max_workers=24) as ex:
    futs = {ex.submit(mapbox, a, b, t): (hh, key)
            for (hh, key, a, b, t) in jobs}
    for f in list(futs):
        hh, key = futs[f]
        state["times"][hh][key] = f.result()
        done += 1
        if done % 500 == 0:
            json.dump(state, open(OUT, "w"))
            print(f"{done}/{len(jobs)} (saved)")
json.dump(state, open(OUT, "w"))
miss = sum(1 for v in state["times"].values() for x in v.values() if x is None)
print(f"DONE: {done} requests, {miss} failed lookups, wrote {OUT}")
