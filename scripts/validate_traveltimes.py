"""Sanity-check: do shortest-path travel times through OUR calibrated speed
field (the distances the warp tries to match) agree with real Mapbox Directions
times for the same origin-destination pairs and hours?

For N random O-D pairs at a few hours: Dijkstra time on our per-edge speeds vs
Mapbox driving-traffic depart_at duration. Reports the ratio (mapbox/ours) per
pair and its spread — a stable ~1.2-1.4x is the expected 'graph runs fast'
offset (no turn penalties); wild or huge ratios mean the speed field is wrong.

Usage: validate_traveltimes.py city graph.graphml [npairs]
"""
import json
import pathlib
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np
import osmnx as ox

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
CITY, GRAPH = sys.argv[1], sys.argv[2]
NPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 18
HOURS = [4, 17]                       # off-peak, PM peak
DATE = "2026-06-15"
CATS = ["low", "moderate", "heavy", "severe"]

cal = json.load(open(ROOT / "data" / f"{CITY}_calib.json"))
FF, R, pooled = cal["FF_kmh"], cal["ratio"], cal["pooled_r"]
cong = json.load(open(ROOT / "data" / f"{CITY}_edge_cong.json"))
G = ox.load_graphml(GRAPH)

def cls_of(d):
    h = d.get("highway", "?")
    return h[0] if isinstance(h, list) else h

def ratio(cls, sev):
    cat = CATS[sev] if sev >= 0 else "low"
    rr = R.get(cls)
    if rr and rr.get(cat) is not None:
        return rr[cat]
    return pooled.get(cat, 0.6)

# per-edge category by hour (aligned to edge_order)
cat_by = {}
for ei, (u, v, k) in enumerate(cong["edge_order"]):
    cat_by[(u, v, k)] = ei

def set_weights(h):
    cats = cong["cong"][str(h)]
    for u, v, k, d in G.edges(keys=True, data=True):
        cls = cls_of(d)
        ei = cat_by.get((u, v, k))
        sev = cats[ei] if ei is not None else -1
        spd = FF.get(cls, 40.0) * ratio(cls, int(sev))      # kph (class-FF model)
        G[u][v][k]["w"] = float(d["length"]) / max(spd / 3.6, 1.0)  # seconds

nodes = list(G.nodes)
xy = {n: (float(G.nodes[n]["x"]), float(G.nodes[n]["y"])) for n in nodes}
rng = np.random.default_rng(3)
# pick O-D pairs 2.5-9 km apart (great-circle-ish in deg * 111km)
pairs = []
tries = 0
while len(pairs) < NPAIRS and tries < 4000:
    tries += 1
    a, b = rng.choice(len(nodes), 2, replace=False)
    na, nb = nodes[a], nodes[b]
    dkm = np.hypot(xy[na][0]-xy[nb][0], (xy[na][1]-xy[nb][1]))*111*0.8
    if 2.5 <= dkm <= 9:
        pairs.append((na, nb, dkm))

def mapbox(o, d, h):
    coords = f"{xy[o][0]},{xy[o][1]};{xy[d][0]},{xy[d][1]}"
    params = {"access_token": TOKEN, "overview": "false",
              "depart_at": f"{DATE}T{h:02d}:00"}
    url = (f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
           f"{urllib.parse.quote(coords)}?{urllib.parse.urlencode(params)}")
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                j = json.loads(r.read().decode())
            if j.get("routes"):
                return j["routes"][0]["duration"]
        except Exception:
            import time; time.sleep(2)
    return None

for h in HOURS:
    set_weights(h)
    rows = []
    def one(p):
        o, d, dkm = p
        try:
            mine = nx.shortest_path_length(G, o, d, weight="w")
        except Exception:
            return None
        mb = mapbox(o, d, h)
        if mb is None or mine <= 0:
            return None
        return (dkm, mine, mb, mb / mine)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(one, pairs):
            if r:
                rows.append(r)
    ratios = np.array([r[3] for r in rows])
    print(f"\n=== {CITY}  {h:02d}:00  ({len(rows)} pairs) ===")
    print(f"  {'dist_km':>7s} {'ours_min':>8s} {'mapbox_min':>10s} {'ratio':>6s}")
    for dkm, mine, mb, rt in sorted(rows):
        print(f"  {dkm:7.1f} {mine/60:8.1f} {mb/60:10.1f} {rt:6.2f}")
    print(f"  --> mapbox/ours ratio: median {np.median(ratios):.2f}  "
          f"mean {ratios.mean():.2f}  range {ratios.min():.2f}-{ratios.max():.2f}")
    print(f"  (consistent ratio => speeds OK up to a scale; wild spread => bad)")
