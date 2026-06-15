"""Harvest continuous per-segment speeds via Mapbox Directions annotations.

depart_at TYPICAL traffic for all 24 hours of an upcoming weekday — no live
polling needed (typical-traffic is a prediction, queryable now), and it matches
the original Uber data, which was itself a typical-weekday average. Each call
returns ~150-250 per-segment speeds along a coverage route; we save them as
located (lon,lat,kmh) points for mapping onto the graph. Resumable.

Usage: collect_directions.py city graph.graphml [n_routes]
"""
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import osmnx as ox
import pyproj

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
CITY, GRAPH = sys.argv[1], sys.argv[2]
N_ROUTES = int(sys.argv[3]) if len(sys.argv) > 3 else 180
N_ANCHOR = 90
DATE = "2026-06-15"                      # a Monday
OUT = ROOT / "data" / f"{CITY}_dir_links.json"

G = ox.load_graphml(GRAPH)
crs = G.graph.get("crs", "epsg:4326")
nodes = list(G.nodes)
P = np.array([[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in nodes])
to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
lon, lat = to_ll.transform(P[:, 0], P[:, 1])
LL = np.column_stack([lon, lat])         # node lon/lat for the API
projected = np.abs(P).max() > 1000       # UTM vs degrees


def fps(k):
    idx = [int(np.argmax(P[:, 1]))]
    d = np.full(len(P), np.inf)
    for _ in range(k - 1):
        last = P[idx[-1]]
        d = np.minimum(d, np.hypot(P[:, 0] - last[0], P[:, 1] - last[1]))
        idx.append(int(np.argmax(d)))
    return idx


anchors = fps(N_ANCHOR)
A = np.array(anchors)
pairs = set()
for i in anchors:
    dd = np.hypot(P[A, 0] - P[i, 0], P[A, 1] - P[i, 1])
    order = [anchors[j] for j in np.argsort(dd)]
    # favor near-neighbor hops (force Directions onto local streets, not just
    # the same highways) plus a couple mid + one long route per anchor
    picks = order[1:6] + order[len(order) // 3: len(order) // 3 + 2] + order[-1:]
    for j in picks:
        if j != i:
            pairs.add(tuple(sorted((i, j))))
pairs = sorted(pairs)[:N_ROUTES]
print(f"{len(pairs)} routes x 24h = {len(pairs)*24} calls "
      f"(graph {'projected' if projected else 'lonlat'})", flush=True)

done = {}
if OUT.exists():
    done = json.load(open(OUT)).get("links", {})
    print(f"resuming: {len(done)} (route,hour) cells already harvested", flush=True)


def directions(o, d, depart):
    coords = f"{o[0]},{o[1]};{d[0]},{d[1]}"
    params = {"access_token": TOKEN,
              "annotations": "distance,duration,congestion", "overview": "full",
              "geometries": "geojson", "steps": "false", "depart_at": depart}
    url = (f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
           f"{urllib.parse.quote(coords)}?{urllib.parse.urlencode(params)}")
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                j = json.loads(r.read().decode())
            if j.get("code") == "Ok" and j.get("routes"):
                return j["routes"][0]
            return None
        except Exception:
            time.sleep(2)
    return None


def collapse(route):
    # depart_at congestion lives in DURATION, not the speed annotation. Save
    # every segment as (lon, lat, dist_m, dur_s) for full spatial resolution;
    # edge-mapping sums dist & dur per edge, so speed = sum_dist/sum_dur, which
    # cancels the per-segment duration-allocation noise while keeping coverage.
    a = route["legs"][0]["annotation"]
    geom = route["geometry"]["coordinates"]
    dist, dur = a["distance"], a["duration"]
    out = []
    for i in range(len(dist)):
        mlon = (geom[i][0] + geom[i + 1][0]) / 2
        mlat = (geom[i][1] + geom[i + 1][1]) / 2
        out.append([round(mlon, 5), round(mlat, 5),
                    round(dist[i], 1), round(dur[i], 2)])
    return out


jobs = [(ri, h) for ri in range(len(pairs)) for h in range(24)
        if f"{ri}_{h}" not in done]
print(f"{len(jobs)} calls to make", flush=True)


def work(job):
    ri, h = job
    i, j = pairs[ri]
    rt = directions(LL[i], LL[j], f"{DATE}T{h:02d}:00")
    return f"{ri}_{h}", (collapse(rt) if rt else [])


def save():
    json.dump({"date": DATE, "n_routes": len(pairs), "links": done},
              open(OUT, "w"))


cnt = fails = 0
with ThreadPoolExecutor(max_workers=24) as ex:
    futs = {ex.submit(work, j): j for j in jobs}
    for fut in as_completed(futs):
        key, links = fut.result()
        done[key] = links
        if not links:
            fails += 1
        cnt += 1
        if cnt % 300 == 0:
            save()
            nlinks = sum(len(v) for v in done.values())
            print(f"  {cnt}/{len(jobs)} calls, {nlinks} speed points, "
                  f"{fails} empty", flush=True)
save()
nlinks = sum(len(v) for v in done.values())
print(f"done: {cnt} calls, {nlinks} located speed points, {fails} empty",
      flush=True)
print(f"saved {OUT}")
