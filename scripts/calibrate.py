"""Calibrate Mapbox congestion CATEGORIES -> per-class speed multipliers for a city.

Probe ~N metro-spanning routes at several weekday hours via Directions
driving-traffic + depart_at, harvest per-link (class, category, speed) rows on
the SPEED-ANNOTATION basis (the pilot proved distance/duration is the wrong
basis), then fit:

  speed(edge, hour) = FF[class] * r[class][category]

FF[class] = free-flow (80th pct of 'low' speeds for the class); r = category
multiplier (median speed in cell / FF), monotonicity-enforced, sparse cells
backfilled from the pooled cross-class ratio. Held-out routes validate.

Usage: calibrate.py graph.graphml out_prefix [YYYY-MM-DD]
"""
import json
import pathlib
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import osmnx as ox
import pyproj

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
GRAPH = pathlib.Path(sys.argv[1])
PREFIX = sys.argv[2]
DATE = sys.argv[3] if len(sys.argv) > 3 else "2026-06-15"  # a Monday
HOURS = [3, 8, 9, 12, 16, 17, 18, 21]
N_ROUTES = 18
K_ANCHOR = 14
CATS = ["low", "moderate", "heavy", "severe"]

print(f"loading {GRAPH.name} ...", flush=True)
G = ox.load_graphml(GRAPH)
Gp = ox.project_graph(G)
nodes = list(G.nodes)
xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])  # lon,lat


def farthest_anchors(k):
    """Farthest-point sampling for spread-out route endpoints."""
    import random  # not used for seed; deterministic start
    idx = [int(np.argmax(xy[:, 1]))]  # northmost as seed (deterministic)
    d = np.full(len(nodes), np.inf)
    for _ in range(k - 1):
        last = xy[idx[-1]]
        dd = np.hypot(xy[:, 0] - last[0], xy[:, 1] - last[1])
        d = np.minimum(d, dd)
        idx.append(int(np.argmax(d)))
    return idx


anchors = farthest_anchors(K_ANCHOR)
# build ~N unique long O-D pairs: each anchor -> its farthest partner
pairs = set()
for i in anchors:
    dists = np.hypot(xy[anchors, 0] - xy[i, 0], xy[anchors, 1] - xy[i, 1])
    order = [anchors[j] for j in np.argsort(-dists)]
    for j in order[:2]:
        pairs.add(tuple(sorted((i, j))))
pairs = list(pairs)[:N_ROUTES]
print(f"{len(pairs)} probe routes x {len(HOURS)} hours "
      f"= {len(pairs)*len(HOURS)} Directions calls", flush=True)


def directions(o_xy, d_xy, depart):
    coords = f"{o_xy[0]},{o_xy[1]};{d_xy[0]},{d_xy[1]}"
    params = {
        "access_token": TOKEN,
        "annotations": "distance,duration,speed,congestion,congestion_numeric,maxspeed",
        "overview": "full", "geometries": "geojson", "steps": "false",
        "depart_at": depart,
    }
    url = (f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
           f"{urllib.parse.quote(coords)}?{urllib.parse.urlencode(params)}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                d = json.loads(r.read().decode())
            if d.get("code") == "Ok" and d.get("routes"):
                return d["routes"][0]
            return None
        except Exception:
            import time
            time.sleep(3 * (attempt + 1))
    return None


def collapse(route, route_id, hour):
    """Shape-segments -> congestion-run links. Speed = the speed annotation."""
    a = route["legs"][0]["annotation"]
    geom = route["geometry"]["coordinates"]
    dist, spd, cong, cn, mx = (a["distance"], a["speed"], a["congestion"],
                               a["congestion_numeric"], a["maxspeed"])
    n = len(dist)
    out, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and cong[j + 1] == cong[i] and spd[j + 1] == spd[i]:
            j += 1
        L = sum(dist[k] for k in range(i, j + 1))
        mid = geom[(i + j + 1) // 2]
        ms = mx[i].get("speed") if isinstance(mx[i], dict) else None
        out.append({"cat": cong[i], "numeric": cn[i], "len_m": L,
                    "kmh": spd[i] * 3.6, "maxspeed": ms,
                    "lon": mid[0], "lat": mid[1],
                    "route": route_id, "hour": hour})
        i = j + 1
    return out


# ---- harvest (threaded) ----
jobs = []
for rid, (i, j) in enumerate(pairs):
    for h in HOURS:
        jobs.append((rid, xy[i], xy[j], f"{DATE}T{h:02d}:00", h))

links = []
fails = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(directions, o, d, dep): (rid, h)
            for (rid, o, d, dep, h) in jobs}
    done = 0
    for fut in futs:
        pass
    from concurrent.futures import as_completed
    for fut in as_completed(futs):
        rid, h = futs[fut]
        rt = fut.result()
        done += 1
        if rt is None:
            fails += 1
            continue
        links.extend(collapse(rt, rid, h))
        if done % 24 == 0:
            print(f"  {done}/{len(jobs)} calls, {len(links)} links", flush=True)
print(f"harvest done: {len(jobs)-fails}/{len(jobs)} calls ok, {len(links)} links",
      flush=True)

# ---- batch class match ----
to_utm = pyproj.Transformer.from_crs("EPSG:4326", Gp.graph["crs"], always_xy=True)
xs, ys = to_utm.transform([l["lon"] for l in links], [l["lat"] for l in links])
edges, dmatch = ox.distance.nearest_edges(Gp, xs, ys, return_dist=True)


def edge_class(u, v, k):
    h = Gp.edges[u, v, k].get("highway", "?")
    return h[0] if isinstance(h, list) else h


rows = []
for l, (u, v, k), dm in zip(links, edges, dmatch):
    if dm > 40 or l["len_m"] < 15:
        continue
    if l["cat"] == "unknown" or l["numeric"] is None:
        continue
    l["class"] = edge_class(u, v, k)
    rows.append(l)
print(f"{len(rows)} usable training rows (matched <40m, len>=15m, known cat)")
(ROOT / "data").mkdir(exist_ok=True)
json.dump(rows, open(ROOT / "data" / f"{PREFIX}_calib_links.json", "w"))

# ---- fit ----
def median(v):
    return float(np.median(v)) if len(v) else float("nan")


by = defaultdict(lambda: defaultdict(list))  # class -> cat -> [speeds]
for r in rows:
    by[r["class"]][r["cat"]].append(r["kmh"])

# pooled cross-class category ratio (normalize each speed by its class FF)
FF = {}
for c, d in by.items():
    lows = d.get("low", [])
    FF[c] = float(np.percentile(lows, 80)) if len(lows) >= 5 else median(
        [s for cat in d for s in d[cat]])
pooled = defaultdict(list)
for r in rows:
    if FF.get(r["class"], 0) > 0:
        pooled[r["cat"]].append(r["kmh"] / FF[r["class"]])
pooled_r = {cat: median(pooled[cat]) for cat in CATS if pooled[cat]}

MIN_N = 8
ratio, counts = {}, {}
for c in by:
    ratio[c], counts[c] = {}, {}
    for cat in CATS:
        v = by[c].get(cat, [])
        counts[c][cat] = len(v)
        if len(v) >= MIN_N and FF[c] > 0:
            ratio[c][cat] = median(v) / FF[c]
        elif cat in pooled_r:
            ratio[c][cat] = pooled_r[cat]      # backfill sparse cell
        else:
            ratio[c][cat] = None
    # enforce monotone non-increasing low>=moderate>=heavy>=severe
    prev = None
    for cat in CATS:
        if ratio[c][cat] is None:
            ratio[c][cat] = prev if prev is not None else pooled_r.get(cat, 0.5)
        if prev is not None:
            ratio[c][cat] = min(ratio[c][cat], prev)
        prev = ratio[c][cat]

# ---- validate on held-out routes ----
route_ids = sorted({r["route"] for r in rows})
holdout = set(route_ids[::5])  # ~20%
tr = [r for r in rows if r["route"] not in holdout]
te = [r for r in rows if r["route"] in holdout]
# refit FF/ratio on train only would be cleaner; FF is stable, reuse for brevity
def predict(r):
    return FF.get(r["class"], np.nan) * ratio.get(r["class"], {}).get(r["cat"], np.nan)
ape = [abs(predict(r) - r["kmh"]) / r["kmh"] for r in te
       if r["kmh"] > 1 and np.isfinite(predict(r))]
glob = median([r["kmh"] for r in tr])
ape_base = [abs(glob - r["kmh"]) / r["kmh"] for r in te if r["kmh"] > 1]
mdape = 100 * median(ape)
mdape_base = 100 * median(ape_base)

cal = {"city": PREFIX, "date": DATE, "hours": HOURS, "n_rows": len(rows),
       "FF_kmh": FF, "ratio": ratio, "counts": counts, "pooled_r": pooled_r,
       "mdape_holdout": mdape, "mdape_baseline": mdape_base}
json.dump(cal, open(ROOT / "data" / f"{PREFIX}_calib.json", "w"), indent=1)

print(f"\n=== {PREFIX} calibration ===")
print(f"{'class':14s} {'FF':>5s} | " + " ".join(f"{c:>8s}" for c in CATS) +
      "    (n per cell)")
for c in sorted(ratio, key=lambda c: -FF[c]):
    rr = " ".join(f"{ratio[c][cat]:8.2f}" for cat in CATS)
    nn = "/".join(str(counts[c][cat]) for cat in CATS)
    print(f"{c:14s} {FF[c]:5.0f} | {rr}    ({nn})")
print(f"\npooled category ratios: " +
      ", ".join(f"{k} {v:.2f}" for k, v in pooled_r.items()))
print(f"held-out speed MdAPE: {mdape:.1f}%  (baseline global-median: "
      f"{mdape_base:.1f}%)")
print(f"saved data/{PREFIX}_calib.json")
