"""Compose the overlay (title, clock, time, yardstick) onto cached base
frames and encode. Fast (~30-45 s) — iterate on LAYOUT freely; only
render_base.py is expensive.

Usage: compose_video.py [suffix]     (frames from data/frames<suffix>/)
Writes shots/breathing<suffix>.mp4
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- layout --
# All values in output pixels (1080 x 1800 frame).
LAYOUT = dict(
    title_xy=(59, 68),            # top-left of "Manhattan"
    title_px=71,                  # font height

    clock_cx=958,                 # clock center x  (was 938; +1 char nudge)
    clock_cy=128,                 # clock center y
    clock_r=68,                   # face radius
    clock_ring_w=4,               # ring stroke
    clock_tick_w=(4, 3),          # cardinal, minor tick strokes
    clock_hand_w=7,               # hour hand stroke
    clock_hand_len=0.56,          # fraction of radius
    clock_dot_r=7,                # center dot radius

    time_gap=9,                   # half-gap between digits and AM/PM
    time_y=222,                   # top of time text
    time_px=54,                   # font height
    # the digits/AM-PM alignment spine is derived from font metrics so the
    # average visual center of all 24 hour-strings sits exactly at clock_cx

    yard_bar_y=1695,              # yardstick bar y
    yard_tick_h=17,               # half-height of end ticks
    yard_w=4,                     # bar stroke
    yard_label_y=1722,            # top of label
    yard_label_px=56,             # font height

    fg="#e8eef6", dim="#a8b8ca", bar="#7d90a8",
    ring="#2a3a4f", tick_card="#7d90a8", tick_min="#3a4a60",
    face="#0a0e16", dot="#e8814a",
)
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
TITLE = sys.argv[2] if len(sys.argv) > 2 else "Manhattan"
# travel-time calibration vs external routers (ours run fast: no turn or
# signal penalties at intersections). Divides the yardstick's reach.
TIME_SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
FRAMES = ROOT / "data" / f"frames{SUFFIX}"
meta = json.load(open(FRAMES / "meta.json"))
W, H = meta["width"], meta["height"]
STEPS = meta["steps"]

TIMES = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
TIMES_I = "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf"
f_title = ImageFont.truetype(TIMES, LAYOUT["title_px"])
f_time = ImageFont.truetype(TIMES, LAYOUT["time_px"])
f_yard = ImageFont.truetype(TIMES_I, LAYOUT["yard_label_px"])

# derive the time alignment spine: with digits right-aligned at S-gap and
# AM/PM left-aligned at S+gap, an hour-string's visual center is
# S + (w_ampm - w_digits)/2. Choose S so the average over all 24 hours
# lands exactly on the clock axis. (Times digits are tabular, so the only
# width classes are 1- vs 2-digit hours and AM vs PM; residual wobble ~7px.)
w1 = f_time.getlength("4")
w2 = f_time.getlength("12")
mean_wd = (18 * w1 + 6 * w2) / 24      # hours 1-9 x2, 10-12 x2
mean_wa = (f_time.getlength("AM") + f_time.getlength("PM")) / 2
TIME_SPLIT_X = LAYOUT["clock_cx"] - (mean_wa - mean_wd) / 2

# ---- static overlay: title + yardstick (drawn once) ----
static = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(static)
d.text(LAYOUT["title_xy"], TITLE, font=f_title, fill=LAYOUT["fg"])

yard_px = meta["yard10_m"] * meta["px_per_m"] / TIME_SCALE
xa, xb = W / 2 - yard_px / 2, W / 2 + yard_px / 2
yy = LAYOUT["yard_bar_y"]
d.line([(xa, yy), (xb, yy)], fill=LAYOUT["bar"], width=LAYOUT["yard_w"])
for xx in (xa, xb):
    d.line([(xx, yy - LAYOUT["yard_tick_h"]), (xx, yy + LAYOUT["yard_tick_h"])],
           fill=LAYOUT["bar"], width=LAYOUT["yard_w"])
d.text((W / 2, LAYOUT["yard_label_y"]), "≈ 10 min drive", font=f_yard,
       fill=LAYOUT["dim"], anchor="ma")

# ---- clock face (static part), supersampled 4x ----
SS = 4
R = LAYOUT["clock_r"]
PATCH = int(R * 2.4)
face = Image.new("RGBA", (PATCH * SS, PATCH * SS), (0, 0, 0, 0))
fd = ImageDraw.Draw(face)
c = PATCH * SS / 2
fd.ellipse([c - R*SS, c - R*SS, c + R*SS, c + R*SS],
           fill=LAYOUT["face"], outline=LAYOUT["ring"],
           width=LAYOUT["clock_ring_w"] * SS)
for k in range(12):
    ang = k * np.pi / 6
    card = k % 3 == 0
    r1 = (0.74 if card else 0.84) * R * SS
    r2 = 0.93 * R * SS
    w = (LAYOUT["clock_tick_w"][0] if card else LAYOUT["clock_tick_w"][1]) * SS
    col = LAYOUT["tick_card"] if card else LAYOUT["tick_min"]
    fd.line([(c + r1*np.sin(ang), c - r1*np.cos(ang)),
             (c + r2*np.sin(ang), c - r2*np.cos(ang))], fill=col, width=w)

def clock_patch(tau):
    """Face + hand + dot for time tau, downscaled to PATCH px."""
    img = face.copy()
    pd = ImageDraw.Draw(img)
    ang = (tau % 12) / 12 * 2 * np.pi
    r = LAYOUT["clock_hand_len"] * R * SS
    pd.line([(c, c), (c + r*np.sin(ang), c - r*np.cos(ang))],
            fill=LAYOUT["fg"], width=LAYOUT["clock_hand_w"] * SS)
    dr = LAYOUT["clock_dot_r"] * SS
    pd.ellipse([c - dr, c - dr, c + dr, c + dr], fill=LAYOUT["dot"])
    return img.resize((PATCH, PATCH), Image.LANCZOS)

# ---- compose + encode ----
mp4 = ROOT / "shots" / f"breathing{SUFFIX}.mp4"
ff = subprocess.Popen(
    ["ffmpeg", "-y", "-loglevel", "error",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
     "-framerate", "48", "-i", "-",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
     "-movflags", "+faststart", str(mp4)],
    stdin=subprocess.PIPE)

ccx, ccy = LAYOUT["clock_cx"], LAYOUT["clock_cy"]
for s in range(STEPS):
    tau = (meta["tau0"] + s / meta["frames_per_hour"]) % 24
    frame = Image.open(FRAMES / f"f{s:04d}.png").convert("RGBA")
    frame.alpha_composite(static)
    patch = clock_patch(tau)
    frame.alpha_composite(patch, (int(ccx - PATCH/2), int(ccy - PATCH/2)))

    fd2 = ImageDraw.Draw(frame)
    hh = int(tau)
    h12 = 12 if hh % 12 == 0 else hh % 12
    fd2.text((TIME_SPLIT_X - LAYOUT["time_gap"], LAYOUT["time_y"]),
             str(h12), font=f_time, fill=LAYOUT["fg"], anchor="ra")
    fd2.text((TIME_SPLIT_X + LAYOUT["time_gap"], LAYOUT["time_y"]),
             "AM" if hh < 12 else "PM", font=f_time, fill=LAYOUT["fg"],
             anchor="la")

    ff.stdin.write(frame.convert("RGB").tobytes())
    if s % 108 == 0:
        print(f"compose {s}/{STEPS}")

ff.stdin.close()
ff.wait()
print(mp4.name, mp4.stat().st_size // 1024, "KB")
