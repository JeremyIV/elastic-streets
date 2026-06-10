"""Prototype: can a sparse anchor grid + Mapbox depart_at typical-traffic
times reproduce direct long-range drive times via offset-corrected
path sums? Validates the cheap-API-city concept before a full 24h sweep.

Usage: anchor_probe.py  (reads data/la_slice.graphml, writes data/la_probe.json)
"""

import json
import pathlib
import sys
import time
import urllib.request

import numpy as np
import osmnx as ox
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
K = 50
EDGE_MAX_M = 4500
HOURS = {"04": "2026-06-17T04:00", "17": "2026-06-17T17:00"}

print("loading graph...")
G = ox.load_graphml(ROOT / "data" / "la_slice.graphml")
Gp = ox.project_graph(G)

ART = {"motorway", "trunk", "primary", "secondary",
       "motorway_link", "trunk_link", "primary_link"}
FWY = {"motorway"}

def edge_hwys(d):
    hw = d.get("highway")
    return hw if isinstance(hw, list) else [hw]

art_nodes, fwy_nodes = set(), set()
for u, v, d in Gp.edges(data=True):
    hs = edge_hwys(d)
    if any(h in ART for h in hs):
        art_nodes.add(u); art_nodes.add(v)
    if any(h in FWY for h in hs):
        fwy_nodes.add(u); fwy_nodes.add(v)
art_nodes = sorted(art_nodes)
print(f"{len(art_nodes)} arterial nodes, {len(fwy_nodes)} freeway nodes")

import networkx as nx
# freeway anchors: farthest-point sample along the motorway subgraph,
# chained by network distance so fast corridors exist as graph paths
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
while fd.max() > 3000:
    j = int(fd.argmax())
    fsel.append(j)
    fd = np.minimum(fd, np.linalg.norm(FXY - FXY[j], axis=1))
fanchors = [fn[i] for i in fsel]
print(f"{len(fanchors)} freeway anchors")

fwy_edges = set()
for a in range(len(fanchors)):
    dists = nx.single_source_dijkstra_path_length(
        Gf, fanchors[a], cutoff=5500, weight="w")
    for b in range(len(fanchors)):
        if a < b and fanchors[b] in dists:
            fwy_edges.add((a, b))
print(f"{len(fwy_edges)} freeway chain edges")

# surface anchors fill the rest, kept off the freeway anchors' turf
XY = np.array([[Gp.nodes[n]["x"], Gp.nodes[n]["y"]] for n in art_nodes])
sel_nodes = list(fanchors)
axy_list = [[Gp.nodes[n]["x"], Gp.nodes[n]["y"]] for n in fanchors]
dmin = np.min(np.linalg.norm(
    XY[:, None, :] - np.array(axy_list)[None, :, :], axis=2), axis=1)
while len(sel_nodes) < K + len(fanchors):
    j = int(dmin.argmax())
    sel_nodes.append(art_nodes[j])
    axy_list.append([XY[j][0], XY[j][1]])
    dmin = np.minimum(dmin, np.linalg.norm(XY - XY[j], axis=1))
axy = np.array(axy_list)
all_ = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in sel_nodes])
K = len(sel_nodes)
print(f"{K} anchors total, min spacing {dmin.max():.0f} m floor")

# connectivity: Delaunay fabric + freeway chains
tri = Delaunay(axy)
edges = set(fwy_edges)
for s in tri.simplices:
    for a in range(3):
        i, j = sorted((int(s[a]), int(s[(a + 1) % 3])))
        if np.linalg.norm(axy[i] - axy[j]) < EDGE_MAX_M:
            edges.add((i, j))
edges = sorted(edges)
print(f"{len(edges)} edges (Delaunay < {EDGE_MAX_M} m + freeway chains)")

# direct validation pairs: long-range, well separated
rng = np.random.default_rng(7)
cand = [(i, j) for i in range(K) for j in range(i + 1, K)
        if np.linalg.norm(axy[i] - axy[j]) > 12000]
val_pairs = [tuple(int(x) for x in cand[i])
             for i in rng.choice(len(cand), 14, replace=False)]

def mapbox(i, j, t):
    time.sleep(1.0)   # 5 workers x 1s -> ~5 req/s, inside the 300/min cap
    url = (f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
           f"{all_[i][0]:.6f},{all_[i][1]:.6f};{all_[j][0]:.6f},{all_[j][1]:.6f}"
           f"?access_token={TOKEN}&depart_at={t}&overview=false")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                d = json.load(r)
            if d.get("routes"):
                return d["routes"][0]["duration"]
            return None
        except Exception as e:
            if attempt == 3:
                print("FAIL", i, j, e)
                return None
            time.sleep(2 * (attempt + 1))

from concurrent.futures import ThreadPoolExecutor

jobs = []
for h, t in HOURS.items():
    for (i, j) in edges:
        jobs.append((h, f"{i}-{j}", i, j, t))
        jobs.append((h, f"{j}-{i}", j, i, t))
    for (i, j) in val_pairs:
        jobs.append((h, f"V{i}-{j}", i, j, t))

results = {h: {} for h in HOURS}
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(mapbox, a, b, t): (h, key)
            for (h, key, a, b, t) in jobs}
    done = 0
    for f in list(futs):
        h, key = futs[f]
        results[h][key] = f.result()
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(jobs)} requests...")
print(f"{len(jobs)} total requests")

json.dump({"anchors_ll": all_.tolist(), "anchors_xy": axy.tolist(),
           "edges": [list(e) for e in edges],
           "val_pairs": [list(p) for p in val_pairs],
           "times": results},
          open(ROOT / "data" / "la_probe.json", "w"))

# ---- offset fit + validation ----
for h in HOURS:
    tm = results[h]
    ew = {}
    for (i, j) in edges:
        ts = [tm.get(f"{i}-{j}"), tm.get(f"{j}-{i}")]
        ts = [t for t in ts if t]
        if ts:
            ew[(i, j)] = np.mean(ts)
    ii = np.array([e[0] for e in ew]); jj = np.array([e[1] for e in ew])
    ww = np.array(list(ew.values()))

    def pathsum(c):
        w = np.maximum(ww - c, 30.0)
        adj = coo_matrix((np.r_[w, w], (np.r_[ii, jj], np.r_[jj, ii])),
                         shape=(K, K)).tocsr()
        D = dijkstra(adj, indices=[p[0] for p in val_pairs])
        return np.array([D[k, j] + c for k, (i, j) in enumerate(val_pairs)])

    direct = np.array([tm.get(f"V{i}-{j}") or np.nan for (i, j) in val_pairs])
    best = None
    for c in range(0, 301, 10):
        ps = pathsum(c)
        err = np.nanmedian(np.abs(ps - direct) / direct)
        if best is None or err < best[1]:
            best = (c, err, ps)
    c, err, ps = best
    print(f"\n=== {h}:00  best offset c={c}s  median |err| {err*100:.1f}%")
    for k, (i, j) in enumerate(val_pairs):
        if np.isfinite(direct[k]):
            print(f"  pair {i:2d}-{j:2d}: direct {direct[k]/60:5.1f}m  "
                  f"pathsum {ps[k]/60:5.1f}m  ({(ps[k]-direct[k])/direct[k]*100:+5.1f}%)")
