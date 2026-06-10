"""Apply an hour-dependent travel-time calibration to a solved day.

The constant calibration (vs OSRM / Google's good-day times) is right
off-peak but understates rush hour: Google's typical 5 PM trip takes ~2x
its 4 AM time, while the 2019 Uber speeds swell only ~1.5x. Fix: scale
each hour's layout about its centroid by cal(h)/cal(4), where cal(h)
interpolates linearly (in the layout's own congestion scale) between the
measured anchors cal(4 AM) and cal(5 PM). Yardstick distances scale with
it, so one constant yardstick is then exact at every hour.

Usage: amplify_breathing.py in.json out.json cal_4am cal_5pm
e.g.   amplify_breathing.py data/day_mds.json data/day_mds_amp.json 1.77 2.37
"""

import json
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
CAL4, CAL17 = float(sys.argv[3]), float(sys.argv[4])

data = json.load(open(src))
N = data["meta"]["n_nodes"]
hours = [np.array(h, float).reshape(N, 2) for h in data["node_hours"]]

# per-hour linear scale of the layout: RMS distance from centroid
s = np.array([np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean()) for P in hours])
cal = CAL4 + (CAL17 - CAL4) * (s - s[4]) / (s[17] - s[4])
g = cal / CAL4

for h, (P, gh) in enumerate(zip(hours, g)):
    c = P.mean(0)
    data["node_hours"][h] = ((P - c) * gh + c).ravel().tolist()
for key in ("yard10", "yard20"):
    if key in data["meta"]:
        data["meta"][key] = [y * gh for y, gh in zip(data["meta"][key], g)]
data["meta"]["cal_curve"] = cal.tolist()
data["meta"]["cal_anchors"] = [CAL4, CAL17]

json.dump(data, open(dst, "w"))
old = (s.max() - s.min()) / s[s.argmin()]
snew = s * g
new = (snew.max() - snew.min()) / snew[snew.argmin()]
print(f"cal(h): " + " ".join(f"{c:.2f}" for c in cal))
print(f"amplify g(h): min {g.min():.3f} (h={g.argmin()}) max {g.max():.3f} (h={g.argmax()})")
print(f"breathing span: {old*100:.0f}% -> {new*100:.0f}%")
print(f"wrote {dst}")
