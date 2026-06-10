"""Render the day animation as a portrait-mode GIF + MP4 (mobile framing)."""

import json
import pathlib
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image

import sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
DAY_FILE = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "day_mds.json")
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
data = json.load(open(DAY_FILE))

N = data["meta"]["n_nodes"]
hours = [np.array(h).reshape(N, 2) for h in data["node_hours"]]
edges = data["edges"]

# color ramp: current speed / edge's fastest hour -> red crawl / blue calm
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
pts = [np.array(e["pts"]).reshape(-1, 2) for e in edges]
zpts = [p[:, 0] + 1j * p[:, 1] for p in pts]
sp = np.array([e["sp"] for e in edges])          # (E, 24)
obs = np.array([e["obs"] for e in edges])        # (E, 24)
eref = sp.max(axis=1)                            # per-edge night pace

# portrait frame. Bounds across all hours, with extra headroom for the
# title/clock band at top and the yardstick at bottom. The x-range is then
# widened to exactly the figure aspect so the framing is identical in every
# frame (adjustable="datalim" would re-derive it per frame and crop things).
ASPECT = 7.2 / 12.0
allp = np.vstack(hours)
PAD = 500
y0 = allp[:, 1].min() - 1650
y1 = allp[:, 1].max() + 2050
xmid = (allp[:, 0].min() + allp[:, 0].max()) / 2
xspan = max((y1 - y0) * ASPECT, allp[:, 0].max() - allp[:, 0].min() + 2 * PAD)
x0, x1 = xmid - xspan / 2, xmid + xspan / 2
y1 += (xspan / ASPECT - (y1 - y0))      # keep aspect exact if x was binding

# yardstick: 10 min of travel, calibrated against the layouts (daily mean
# of the per-hour medians; a ruler shouldn't wiggle with solver misfit).
# Drawn as a figure overlay: length in data units, pinned near the bottom.
yard_len = float(np.mean(data["meta"]["yard10"]))
bar_w = yard_len / xspan
bar_y = 0.046

def catmull_weights(t):
    t2, t3 = t * t, t ** 3
    return (0.5 * (-t + 2*t2 - t3), 0.5 * (2 - 5*t2 + 3*t3),
            0.5 * (t + 4*t2 - 3*t3), 0.5 * (-t2 + t3))

import os
TMP = tempfile.mkdtemp()
STEPS = 432                      # 18 frames/hour -> 48 fps, 9 s loop
for s in range(STEPS):
    tau = (4 + s / 18.0) % 24
    h1 = int(tau) % 24
    h0, h2, h3 = (h1 + 23) % 24, (h1 + 1) % 24, (h1 + 2) % 24
    t = tau - int(tau)
    w0, w1, w2, w3 = catmull_weights(t)
    P = w0*hours[h0] + w1*hours[h1] + w2*hours[h2] + w3*hours[h3]

    a = P[eu][:, 0] + 1j * P[eu][:, 1]
    b = P[ev][:, 0] + 1j * P[ev][:, 1]
    segs, cols = [], []
    spd = np.maximum(w0*sp[:, h0] + w1*sp[:, h1] + w2*sp[:, h2] + w3*sp[:, h3], 1.0)
    ratio = spd / eref
    # observed/imputed dimming, splined like everything else so edges fade
    # smoothly instead of popping at hour boundaries
    vis = np.clip(w0*obs[:, h0] + w1*obs[:, h1] + w2*obs[:, h2] + w3*obs[:, h3],
                  0.0, 1.0)
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

    # title
    fig.text(0.055, 0.957, "Manhattan",
             ha="left", va="top", color="#e8eef6", fontsize=34,
             family="Times New Roman")

    # clock above the numeric time, top-right
    CW = 0.145                       # ~104 px wide, 2x the old face
    CH = CW * 7.2 / 12.0
    ccx = 0.85
    cax = fig.add_axes([ccx - CW / 2, 0.896, CW, CH])
    cax.set_xlim(-1.15, 1.15); cax.set_ylim(-1.15, 1.15)
    cax.set_aspect("equal"); cax.axis("off")
    cax.add_patch(plt.Circle((0, 0), 1.0, fill=True, facecolor="#0a0e16",
                             edgecolor="#2a3a4f", lw=2.0))
    for k in range(12):
        ang = k * np.pi / 6
        cardinal = k % 3 == 0
        r1 = 0.74 if cardinal else 0.84
        cax.plot([r1 * np.sin(ang), 0.93 * np.sin(ang)],
                 [r1 * np.cos(ang), 0.93 * np.cos(ang)],
                 color="#7d90a8" if cardinal else "#3a4a60",
                 lw=2.2 if cardinal else 1.4, solid_capstyle="round")
    hang = (tau % 12) / 12 * 2 * np.pi
    cax.plot([0, 0.56 * np.sin(hang)], [0, 0.56 * np.cos(hang)],
             color="#e8eef6", lw=3.4, solid_capstyle="round")
    cax.add_patch(plt.Circle((0, 0), 0.10, color="#e8814a"))

    # numeric time below the clock: digits right-aligned, AM/PM fixed
    hh = int(tau)
    h12 = 12 if hh % 12 == 0 else hh % 12
    fig.text(ccx - 0.008, 0.886, f"{h12}", ha="right", va="top",
             color="#e8eef6", fontsize=26, family="Times New Roman")
    fig.text(ccx + 0.008, 0.886, "AM" if hh < 12 else "PM", ha="left",
             va="top", color="#e8eef6", fontsize=26, family="Times New Roman")

    # yardstick (figure overlay)
    xa_f, xb_f = 0.5 - bar_w / 2, 0.5 + bar_w / 2
    fig.add_artist(plt.Line2D([xa_f, xb_f], [bar_y, bar_y],
                              transform=fig.transFigure,
                              color="#7d90a8", lw=1.8))
    for xx in (xa_f, xb_f):
        fig.add_artist(plt.Line2D([xx, xx], [bar_y - 0.0095, bar_y + 0.0095],
                                  transform=fig.transFigure,
                                  color="#7d90a8", lw=1.8))
    fig.text(0.5, 0.030, "≈ 10 min drive", ha="center", va="top",
             color="#a8b8ca", fontsize=17, family="Times New Roman", style="italic")

    fig.savefig(f"{TMP}/f{s:04d}.png", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    if s % 36 == 0:
        print(f"frame {s}/{STEPS}")

# MP4: 24 fps, high quality
mp4 = ROOT / "shots" / f"breathing{SUFFIX}.mp4"
subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "48",
     "-i", f"{TMP}/f%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-crf", "17", "-movflags", "+faststart", str(mp4)], check=True)
print(mp4.name, mp4.stat().st_size // 1024, "KB")

# GIF preview: every 9th frame, downscaled
gif_frames = []
for s in range(0, STEPS, 9):
    im = Image.open(f"{TMP}/f{s:04d}.png").convert("RGB")
    gif_frames.append(im.resize((720, int(im.height * 720 / im.width)),
                                Image.LANCZOS))
out = ROOT / "shots" / f"breathing{SUFFIX}.gif"
gif_frames[0].save(out, save_all=True, append_images=gif_frames[1:],
                   duration=140, loop=0)
im = Image.open(out)
print(im.size, "frames:", im.n_frames, out.stat().st_size // 1024, "KB")
for s in range(STEPS):
    os.unlink(f"{TMP}/f{s:04d}.png")
os.rmdir(TMP)
