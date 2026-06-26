"""Diagnostic: plot streets in TRUE geographic position, colored by raw Mapbox
congestion category at several hours. No calibration, no warping — just the
data. Answers: does the tile data actually distinguish night from rush hour, or
is everything "low"?

Usage: congestion_map.py city graph.graphml [W S E N]   (optional bbox clip)
"""
import json
import pathlib
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.collections import LineCollection

ROOT = pathlib.Path(__file__).resolve().parent.parent
CITY = sys.argv[1]
GRAPH = sys.argv[2]
BBOX = [float(x) for x in sys.argv[3:7]] if len(sys.argv) >= 7 else None
HOURS = [4, 8, 17]                         # deep night, AM rush, PM rush
NAMES = {4: "4 AM", 8: "8 AM", 17: "5 PM"}

# category -> (color, linewidth, zorder). Congested drawn on top, bright.
STYLE = {
    0:  ((0.30, 0.42, 0.58), 0.5, 1),      # low      muted blue
    1:  ((1.00, 0.82, 0.10), 1.1, 3),      # moderate yellow
    2:  ((1.00, 0.45, 0.05), 1.4, 4),      # heavy    orange
    3:  ((0.96, 0.12, 0.12), 1.8, 5),      # severe   red
    -1: ((0.22, 0.22, 0.26), 0.4, 0),      # no data  grey
}
LABEL = {0: "low", 1: "moderate", 2: "heavy", 3: "severe", -1: "no data"}

cong = json.load(open(ROOT / "data" / f"{CITY}_edge_cong.json"))
G = ox.load_graphml(GRAPH)
xy = {n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in G.nodes}

# geometry (lon/lat) per edge, aligned to edge_order
geoms, mids = [], []
for u, v, k in cong["edge_order"]:
    d = G.edges[u, v, k]
    if "geometry" in d:
        c = np.array(d["geometry"].coords)
    else:
        c = np.array([xy[u], xy[v]])
    geoms.append(c)
    mids.append(c.mean(0))
mids = np.array(mids)

if BBOX:
    W, S, E, N = BBOX
    keep = ((mids[:, 0] >= W) & (mids[:, 0] <= E) &
            (mids[:, 1] >= S) & (mids[:, 1] <= N))
else:
    keep = np.ones(len(geoms), bool)
idxs = np.flatnonzero(keep)
# aspect for equal-area display at this latitude
latc = mids[keep, 1].mean()
aspect = 1.0 / np.cos(np.radians(latc))

fig, axes = plt.subplots(1, len(HOURS), figsize=(5.2 * len(HOURS), 9.5))
fig.patch.set_facecolor("#05070c")
for ax, h in zip(axes, HOURS):
    cats = np.array(cong["cong"][str(h)])
    counts = Counter(int(cats[i]) for i in idxs)
    tot = len(idxs)
    congested = sum(counts.get(c, 0) for c in (1, 2, 3))
    segs, cols, lws, zs = [], [], [], []
    for i in idxs:
        c = int(cats[i])
        col, lw, z = STYLE[c]
        segs.append(geoms[i])
        cols.append(col); lws.append(lw); zs.append(z)
    order = np.argsort(zs)                  # draw congested last (on top)
    lc = LineCollection([segs[o] for o in order],
                        colors=[cols[o] for o in order],
                        linewidths=[lws[o] for o in order])
    ax.add_collection(lc)
    ax.set_xlim(mids[keep, 0].min(), mids[keep, 0].max())
    ax.set_ylim(mids[keep, 1].min(), mids[keep, 1].max())
    ax.set_aspect(aspect)
    ax.set_facecolor("#05070c"); ax.axis("off")
    pct = lambda c: 100 * counts.get(c, 0) / tot
    title = (f"{NAMES[h]}   congested {100*congested/tot:.0f}%\n"
             f"low {pct(0):.0f}  mod {pct(1):.0f}  "
             f"heavy {pct(2):.0f}  sev {pct(3):.0f}")
    ax.set_title(title, color="#e8eef6", fontsize=13, fontfamily="monospace")
    print(f"{NAMES[h]:6s}: " + "  ".join(
        f"{LABEL[c]} {pct(c):4.1f}%" for c in (0, 1, 2, 3, -1)))

fig.suptitle(f"{CITY} — raw Mapbox congestion category (no warp, no calibration)",
             color="#e8eef6", fontsize=15, y=0.98)
out = ROOT / "shots" / f"congmap_{CITY}.png"
fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
print("wrote", out)
