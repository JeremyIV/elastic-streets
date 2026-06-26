"""Diagnostic VIDEO: streets in true geographic position, colored by raw Mapbox
congestion category, animated across 24 hours. No warping, no calibration — the
honest pulse of the data. Encodes shots/diag_<city>.mp4.

Usage: congestion_anim.py city graph.graphml title [W S E N]
"""
import json
import pathlib
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.collections import LineCollection

ROOT = pathlib.Path(__file__).resolve().parent.parent
CITY, GRAPH, TITLE = sys.argv[1], sys.argv[2], sys.argv[3]
BBOX = [float(x) for x in sys.argv[4:8]] if len(sys.argv) >= 8 else None
FPS, FPH = 24, 12                      # 12 frames/hour -> 288 frames, 12 s loop

# continuous severity (0 low .. 3 severe) -> color
CSTOPS = [(0.0, (0.20, 0.30, 0.46)), (1.0, (0.95, 0.80, 0.15)),
          (2.0, (1.00, 0.45, 0.05)), (3.0, (0.97, 0.10, 0.10))]


def sev_color(s):
    for k in range(1, len(CSTOPS)):
        if s <= CSTOPS[k][0]:
            (v0, c0), (v1, c1) = CSTOPS[k - 1], CSTOPS[k]
            t = (s - v0) / (v1 - v0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return CSTOPS[-1][1]


cong = json.load(open(ROOT / "data" / f"{CITY}_edge_cong.json"))
G = ox.load_graphml(GRAPH)
xy = {n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in G.nodes}
geoms, mids = [], []
for u, v, k in cong["edge_order"]:
    d = G.edges[u, v, k]
    c = np.array(d["geometry"].coords) if "geometry" in d else np.array([xy[u], xy[v]])
    geoms.append(c); mids.append(c.mean(0))
mids = np.array(mids)

keep = np.ones(len(geoms), bool)
if BBOX:
    W, S, E, N = BBOX
    keep = ((mids[:, 0] >= W) & (mids[:, 0] <= E) &
            (mids[:, 1] >= S) & (mids[:, 1] <= N))
idxs = np.flatnonzero(keep)

# severity matrix (E,24); no-data -> low(0)
sev = np.array([[max(0, cong["cong"][str(h)][i]) for h in range(24)]
                for i in idxs], dtype=float)
segs = [geoms[i] for i in idxs]
peak = sev.max(1)
order = np.argsort(peak)               # draw ever-congested streets on top
segs = [segs[o] for o in order]
sev = sev[order]

latc = mids[keep, 1].mean()
aspect = 1.0 / np.cos(np.radians(latc))
x0, x1 = mids[keep, 0].min(), mids[keep, 0].max()
y0, y1 = mids[keep, 1].min(), mids[keep, 1].max()

fig = plt.figure(figsize=(6.0, 10.0), dpi=150)
fig.patch.set_facecolor("#05070c")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("#05070c")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
ax.set_aspect(aspect); ax.axis("off")
lc = LineCollection(segs, linewidths=0.6)
ax.add_collection(lc)
title_t = ax.text(0.04, 0.972, TITLE, transform=ax.transAxes, color="#e8eef6",
                  fontsize=22, family="serif", va="top")
hour_t = ax.text(0.96, 0.972, "", transform=ax.transAxes, color="#e8eef6",
                 fontsize=20, family="serif", ha="right", va="top")
pct_t = ax.text(0.96, 0.945, "", transform=ax.transAxes, color="#8aa0b8",
                fontsize=12, family="monospace", ha="right", va="top")

STEPS = 24 * FPH
tmp = pathlib.Path(tempfile.mkdtemp())
for s in range(STEPS):
    tau = s / FPH
    h0 = int(tau) % 24
    h1 = (h0 + 1) % 24
    t = tau - int(tau)
    sv = sev[:, h0] * (1 - t) + sev[:, h1] * t
    cols = [(*sev_color(v), 0.40 + 0.20 * min(v, 3)) for v in sv]
    lc.set_color(cols)
    lc.set_linewidths(0.5 + 0.55 * sv)
    hh = int(round(tau)) % 24
    ampm = "AM" if hh < 12 else "PM"
    h12 = hh % 12 or 12
    hour_t.set_text(f"{h12} {ampm}")
    congf = 100 * (sv >= 0.5).mean()
    pct_t.set_text(f"{congf:4.1f}% congested")
    fig.savefig(tmp / f"f{s:04d}.png", facecolor=fig.get_facecolor())
    if s % 48 == 0:
        print(f"frame {s}/{STEPS}", flush=True)
plt.close(fig)

out = ROOT / "shots" / f"diag_{CITY}.mp4"
subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                "-movflags", "+faststart", str(out)], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")
