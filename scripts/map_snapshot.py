"""No-warp Seattle congestion snapshot from the CONTINUOUS Directions speeds.

Maps harvested per-segment speeds onto the graph, computes each edge's free-flow
(its fastest hour), and colors streets at chosen hours by speed/free-flow — the
same continuous 'how slow vs its own best' signal the original Uber video used.
This is the test: does continuous speed show more congestion coverage than the
sparse Mapbox category? Prints the covered + congested fractions.

Usage: map_snapshot.py city graph.graphml [hours csv]
"""
import json
import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pyproj
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
CITY, GRAPH = sys.argv[1], sys.argv[2]
HOURS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [4, 8, 17]
NAMES = {4: "4 AM", 8: "8 AM", 12: "noon", 17: "5 PM", 21: "9 PM"}

G = ox.load_graphml(GRAPH)
crs = G.graph.get("crs", "epsg:4326")
to_xy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

edges = list(G.edges(keys=True, data=True))
def midpt(d, u, v):
    if "geometry" in d:
        c = np.array(d["geometry"].coords)
    else:
        c = np.array([[float(G.nodes[u]["x"]), float(G.nodes[u]["y"])],
                      [float(G.nodes[v]["x"]), float(G.nodes[v]["y"])]])
    return c
geoms = [midpt(d, u, v) for u, v, k, d in edges]
emid = np.array([g.mean(0) for g in geoms])
etree = cKDTree(emid)

links = json.load(open(ROOT / "data" / f"{CITY}_dir_links.json"))["links"]
# per-hour: accumulate distance+duration onto nearest edge; speed = sum/sum
E = len(edges)
dacc = np.zeros((E, 24))
tacc = np.zeros((E, 24))
for key, pts in links.items():
    if not pts:
        continue
    h = int(key.split("_")[1])
    arr = np.array(pts)                       # lon,lat,dist_m,dur_s
    X, Y = to_xy.transform(arr[:, 0], arr[:, 1])
    d, j = etree.query(np.column_stack([X, Y]), k=1, distance_upper_bound=40)
    ok = np.isfinite(d)
    for ei, dm, ds in zip(j[ok], arr[ok, 2], arr[ok, 3]):
        dacc[ei, h] += dm
        tacc[ei, h] += ds
sp = np.where(tacc > 0, dacc / np.maximum(tacc, 1e-6) * 3.6, np.nan)

ff = np.nanmax(sp, axis=1)                      # free-flow = fastest hour
covered = np.isfinite(ff)
print(f"coverage: {covered.sum()}/{E} edges = {100*covered.mean():.0f}% "
      f"have continuous speed data")

STOPS = [(0.30, (1.00, 0.20, 0.12)), (0.50, (1.00, 0.45, 0.10)),
         (0.70, (1.00, 0.78, 0.30)), (0.88, (0.55, 0.62, 0.72)),
         (1.00, (0.40, 0.62, 0.90))]
def rcolor(r):
    if r <= STOPS[0][0]:
        return STOPS[0][1]
    for k in range(1, len(STOPS)):
        if r <= STOPS[k][0]:
            (v0, c0), (v1, c1) = STOPS[k - 1], STOPS[k]
            t = (r - v0) / (v1 - v0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return STOPS[-1][1]

fig, axes = plt.subplots(1, len(HOURS), figsize=(5.2 * len(HOURS), 9.5))
fig.patch.set_facecolor("#05070c")
x0, y0 = emid[covered].min(0)
x1, y1 = emid[covered].max(0)
for ax, h in zip(axes, HOURS):
    segs, cols, lws = [], [], []
    cong = 0
    n = 0
    for ei in range(E):
        if not covered[ei] or not np.isfinite(sp[ei, h]):
            continue
        r = np.clip(sp[ei, h] / ff[ei], 0, 1.2)
        segs.append(geoms[ei]); cols.append(rcolor(r))
        lws.append(0.6 + 1.2 * (1 - min(r, 1)))
        n += 1
        if r < 0.7:
            cong += 1
    order = np.argsort([-l for l in lws])
    ax.add_collection(LineCollection([segs[o] for o in order],
                      colors=[cols[o] for o in order],
                      linewidths=[lws[o] for o in order]))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal"); ax.set_facecolor("#05070c"); ax.axis("off")
    pc = 100 * cong / max(n, 1)
    ax.set_title(f"{NAMES.get(h, str(h))}   {pc:.0f}% slowed >30%",
                 color="#e8eef6", fontsize=14, family="monospace")
    print(f"{NAMES.get(h,h):6s}: {pc:.0f}% of covered edges slowed >30% vs free-flow")

fig.suptitle(f"{CITY} — CONTINUOUS speed (Directions), colored vs each road's free-flow",
             color="#e8eef6", fontsize=15, y=0.99)
out = ROOT / "shots" / f"snap_{CITY}_dir.png"
fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
print("wrote", out)
