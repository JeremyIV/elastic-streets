"""Pilot part 2: collapse shape-point segments into road links and reconcile
the two speed bases (distance/duration vs the `speed` annotation). Operates on
the JSON already saved by pilot_calib.py — no API call.

A "link" = a maximal run of consecutive segments sharing the same congestion
category AND speed-annotation value (i.e. one road edge). Per link we get one
calibration observation: (category, congestion_numeric, length, agg speed).
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "data" / "pilot_nyc_route.json").read_text())
ann = data["routes"][0]["legs"][0]["annotation"]
geom = data["routes"][0]["geometry"]["coordinates"]
dist, dur, spd = ann["distance"], ann["duration"], ann["speed"]
cong, cn = ann["congestion"], ann["congestion_numeric"]
mx = ann["maxspeed"]
n = len(dist)

# group consecutive segments sharing (congestion, speed-annotation) into links
links = []
i = 0
while i < n:
    j = i
    while (j + 1 < n and cong[j + 1] == cong[i] and spd[j + 1] == spd[i]):
        j += 1
    seg = range(i, j + 1)
    L = sum(dist[k] for k in seg)
    T = sum(dur[k] for k in seg)
    mid = geom[(i + j + 1) // 2]  # lon,lat midpoint vertex of the link
    ms = mx[i].get("speed") if isinstance(mx[i], dict) else None
    links.append({
        "cat": cong[i], "numeric": cn[i], "len_m": L,
        "kmh_distdur": (L / T * 3.6) if T else 0.0,
        "kmh_ann": spd[i] * 3.6, "maxspeed": ms,
        "lon": mid[0], "lat": mid[1], "nseg": j - i + 1,
    })
    i = j + 1

print(f"{n} shape-segments collapsed to {len(links)} road links")

# reconcile the two speed bases at link level
import statistics
diffs = [abs(l["kmh_distdur"] - l["kmh_ann"]) for l in links
         if l["len_m"] > 20]  # ignore tiny stubs
rel = [abs(l["kmh_distdur"] - l["kmh_ann"]) / max(l["kmh_ann"], 1)
       for l in links if l["len_m"] > 20 and l["kmh_ann"] > 1]
print(f"link-level |distdur - ann| speed: median {statistics.median(diffs):.1f} kmh, "
      f"median rel {100*statistics.median(rel):.0f}%")

from collections import Counter
cc = Counter(l["cat"] for l in links)
print("links by category:", dict(cc))

print(f"\n{'cat':9s} {'numeric':>7s} {'len_m':>7s} {'kmh_dd':>7s} "
      f"{'kmh_ann':>7s} {'maxspd':>6s}  (lon,lat)")
for l in links[:18]:
    print(f"{l['cat']:9s} {str(l['numeric']):>7s} {l['len_m']:7.0f} "
          f"{l['kmh_distdur']:7.1f} {l['kmh_ann']:7.1f} {str(l['maxspeed']):>6s}  "
          f"({l['lon']:.4f},{l['lat']:.4f})")

# stash links for the class-match step
(ROOT / "data" / "pilot_nyc_links.json").write_text(json.dumps(links))
print(f"\nsaved {len(links)} links -> data/pilot_nyc_links.json")
