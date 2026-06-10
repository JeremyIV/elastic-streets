"""Render the expensive layer once: map-only frames to a disk cache.

Usage: render_base.py [day_json] [suffix]
Writes data/frames<suffix>/f%04d.png plus meta.json for compose_video.py.
Only needs re-running when the underlying layouts change.
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

ROOT = pathlib.Path(__file__).resolve().parent.parent
DAY_FILE = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "day_mds.json")
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = ROOT / "data" / f"frames{SUFFIX}"
OUT.mkdir(exist_ok=True)

data = json.load(open(DAY_FILE))
N = data["meta"]["n_nodes"]
hours = [np.array(h).reshape(N, 2) for h in data["node_hours"]]
edges = data["edges"]

STOPS = [(0.30, (1.00, 0.22, 0.13)), (0.52, (1.00, 0.45, 0.20)),
         (0.74, (1.00, 0.72, 0.45)), (0.92, (0.62, 0.66, 0.72)),
         (1.00, (0.45, 0.66, 0.90))]

def ratio_color(r):
    if r <= STOPS[0][0]:
        return STOPS[0][1]
    for k in range(1, len(STOPS)):
        if r <= STOPS[k][0]:
            v0, c0 = STOPS[k - 1]
            v1, c1 = STOPS[k]
            t = (r - v0) / (v1 - v0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return STOPS[-1][1]

eu = np.array([e["u"] for e in edges])
ev = np.array([e["v"] for e in edges])
zpts = [np.array(e["pts"]).reshape(-1, 2) @ np.array([1, 1j]) for e in edges]
sp = np.array([e["sp"] for e in edges])
obs = np.array([e["obs"] for e in edges])
eref = sp.max(axis=1)

# portrait frame, framing identical in every frame
ASPECT = 7.2 / 12.0
allp = np.vstack(hours)
y0 = allp[:, 1].min() - 1650
y1 = allp[:, 1].max() + 2050
xmid = (allp[:, 0].min() + allp[:, 0].max()) / 2
xspan = max((y1 - y0) * ASPECT,
            allp[:, 0].max() - allp[:, 0].min() + 1000)
x0, x1 = xmid - xspan / 2, xmid + xspan / 2
slack = xspan / ASPECT - (y1 - y0)
y0 -= slack / 2                       # center the city when width binds
y1 += slack / 2

# optional zoom (argv[3]): shrink the window around its center; content
# beyond the edges (outlier fringe roads at peak hours) just exits the frame
ZOOM = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
if ZOOM != 1.0:
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = (x1 - x0) / 2 / ZOOM, (y1 - y0) / 2 / ZOOM
    x0, x1, y0, y1 = cx - hx, cx + hx, cy - hy, cy + hy
    xspan = x1 - x0

def catmull_weights(t):
    t2, t3 = t * t, t ** 3
    return (0.5 * (-t + 2*t2 - t3), 0.5 * (2 - 5*t2 + 3*t3),
            0.5 * (t + 4*t2 - 3*t3), 0.5 * (-t2 + t3))

STEPS = 432
W_PX, H_PX = 1080, 1800
for s in range(STEPS):
    tau = (4 + s / 18.0) % 24
    h1 = int(tau) % 24
    h0, h2, h3 = (h1 + 23) % 24, (h1 + 1) % 24, (h1 + 2) % 24
    t = tau - int(tau)
    w0, w1, w2, w3 = catmull_weights(t)
    P = w0*hours[h0] + w1*hours[h1] + w2*hours[h2] + w3*hours[h3]

    a = P[eu][:, 0] + 1j * P[eu][:, 1]
    b = P[ev][:, 0] + 1j * P[ev][:, 1]
    spd = np.maximum(w0*sp[:, h0] + w1*sp[:, h1] + w2*sp[:, h2] + w3*sp[:, h3], 1.0)
    ratio = spd / eref
    vis = np.clip(w0*obs[:, h0] + w1*obs[:, h1] + w2*obs[:, h2] + w3*obs[:, h3],
                  0.0, 1.0)
    segs, cols = [], []
    for i in range(len(edges)):
        z0 = zpts[i]
        d0 = z0[-1] - z0[0]
        if abs(d0) > 1e-9:
            z1 = a[i] + (b[i] - a[i]) / d0 * (z0 - z0[0])
        else:
            z1 = z0 + (a[i] - z0[0])
        segs.append(np.column_stack([z1.real, z1.imag]))
        c = ratio_color(ratio[i])
        cols.append((*c, 0.35 + 0.55 * vis[i]))

    fig = plt.figure(figsize=(7.2, 12.0), dpi=150)
    fig.patch.set_facecolor("#05070c")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#05070c")
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.9))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.savefig(OUT / f"f{s:04d}.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    if s % 36 == 0:
        print(f"frame {s}/{STEPS}")

meta = {
    "steps": STEPS, "width": W_PX, "height": H_PX,
    "tau0": 4.0, "frames_per_hour": 18,
    "px_per_m": W_PX / xspan,
    "yard10_m": float(np.mean(data["meta"]["yard10"])),
}
json.dump(meta, open(OUT / "meta.json", "w"))
print(f"wrote {STEPS} frames + meta to {OUT}")
