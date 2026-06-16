"""Harvest per-segment RESIDENTIAL speeds via the Mapbox Map Matching API.

The Directions API always returns the fastest path, so it rides arterials and
leaves residential streets uncovered. Map Matching is the inverse: you hand it
a trace (a path you choose) and it snaps it to the road network, returning
per-segment distance/duration for THOSE roads. So we walk the drive network
into long traces (<=100 coords, the API cap) that cover every Directions-skipped
local street, and read their typical-traffic speeds back. depart_at gives
24-hour typical traffic, exactly like collect_directions. Output is the same
{city}_match_links.json links format, so solve_directions consumes it directly
(it auto-merges _match_links with _dir_links).

Usage: collect_matching.py city graph.graphml [max_traces]
  max_traces > 0 limits the run (probe mode); 0/omitted = cover all locals.
  Set DRYRUN=1 to print the trace/call plan and exit (no API calls).
"""
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
import numpy as np
import osmnx as ox
import pyproj

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
CITY, GRAPH = sys.argv[1], sys.argv[2]
MAX_TRACES = int(sys.argv[3]) if len(sys.argv) > 3 else 0
DATE = "2026-06-15"                       # a Monday, matches collect_directions
OUT = ROOT / "data" / f"{CITY}_match_links.json"
MAX_PTS = 95                             # Map Matching caps at 100 coords/request

# local classes the Directions router skips (arterials are already covered)
LOCAL = {"residential", "living_street", "unclassified", "service",
         "tertiary", "tertiary_link", "road"}


def cls_of(d):
    h = d.get("highway", "?")
    return h[0] if isinstance(h, list) else h


G = ox.load_graphml(GRAPH)
crs = G.graph.get("crs", "epsg:4326")
ns = list(G.nodes)
X = np.array([float(G.nodes[n]["x"]) for n in ns])
Y = np.array([float(G.nodes[n]["y"]) for n in ns])
to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
LO, LA = to_ll.transform(X, Y)
ll = {n: (round(float(LO[i]), 6), round(float(LA[i]), 6)) for i, n in enumerate(ns)}

# undirected drive network (for connectivity) + the set of local edges to cover
Gu = nx.Graph()
LOCAL_EDGES = set()
for u, v, d in ox.convert.to_undirected(G).edges(data=True):
    Gu.add_edge(u, v)
    if cls_of(d) in LOCAL:
        LOCAL_EDGES.add((u, v) if u < v else (v, u))
print(f"{Gu.number_of_edges()} edges ({len(LOCAL_EDGES)} local) over "
      f"{Gu.number_of_nodes()} nodes", flush=True)


def ek(a, b):
    return (a, b) if a < b else (b, a)


def build_traces(max_pts, max_traces):
    """Cover every LOCAL edge with long traces. When a walk runs out of adjacent
    uncovered local edges it bridges along the shortest path (through any street)
    to the nearest remaining local edge, so each trace packs ~max_pts nodes
    instead of dead-ending after a few blocks. Bridge streets get matched too --
    harmless; they just add (already-covered) arterial samples."""
    remaining = set(LOCAL_EDGES)

    def bridge(cur):
        """BFS in Gu to the nearest node that still touches a remaining edge."""
        rnodes = {x for e in remaining for x in e}
        seen = {cur}
        q = deque([(cur, [cur])])
        while q:
            node, path = q.popleft()
            if node in rnodes and node != cur:
                return path
            for w in Gu.neighbors(node):
                if w not in seen:
                    seen.add(w)
                    q.append((w, path + [w]))
        return None

    traces = []
    while remaining:
        u0, _ = next(iter(remaining))
        chain, cur = [u0], u0
        while len(chain) < max_pts:
            unu = [w for w in Gu.neighbors(cur) if ek(cur, w) in remaining]
            if unu:
                w = unu[0]
                remaining.discard(ek(cur, w))
                chain.append(w)
                cur = w
            else:
                path = bridge(cur)
                if not path:
                    break
                for w in path[1:]:
                    if len(chain) >= max_pts:
                        break
                    remaining.discard(ek(cur, w))
                    chain.append(w)
                    cur = w
        if len(chain) >= 2:
            traces.append(chain)
        if max_traces and len(traces) >= max_traces:
            break
    return traces


traces = build_traces(MAX_PTS, MAX_TRACES)
n_seg = sum(len(t) - 1 for t in traces)
print(f"{len(traces)} traces ({n_seg} segments, avg "
      f"{n_seg / max(len(traces), 1):.0f}/trace) x 24h = {len(traces) * 24} calls",
      flush=True)
if os.environ.get("DRYRUN"):
    sys.exit(0)

done = {}
if OUT.exists():
    done = json.load(open(OUT)).get("links", {})
    print(f"resuming: {len(done)} (trace,hour) cells already harvested", flush=True)


def matching(chain, depart):
    pts = [ll[n] for n in chain]
    coords = ";".join(f"{lo},{la}" for lo, la in pts)
    params = {"access_token": TOKEN,
              "annotations": "distance,duration", "overview": "full",
              "geometries": "geojson", "steps": "false",
              "radiuses": ";".join("25" for _ in pts), "depart_at": depart}
    url = (f"https://api.mapbox.com/matching/v5/mapbox/driving-traffic/"
           f"{urllib.parse.quote(coords)}?{urllib.parse.urlencode(params)}")
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                j = json.loads(r.read().decode())
            if j.get("code") == "Ok" and j.get("matchings"):
                return j["matchings"]
            return None
        except Exception:
            time.sleep(2)
    return None


def collapse(matchings):
    # concatenate per-leg dist/dur, align to the matched geometry segments;
    # save (lon, lat, dist_m, dur_s) midpoints -- identical to collect_directions
    out = []
    for m in matchings:
        geom = m["geometry"]["coordinates"]
        dist, dur = [], []
        for leg in m.get("legs", []):
            a = leg.get("annotation", {})
            dist += a.get("distance", [])
            dur += a.get("duration", [])
        for i in range(min(len(dist), len(dur), len(geom) - 1)):
            mlon = (geom[i][0] + geom[i + 1][0]) / 2
            mlat = (geom[i][1] + geom[i + 1][1]) / 2
            out.append([round(mlon, 5), round(mlat, 5),
                        round(dist[i], 1), round(dur[i], 2)])
    return out


jobs = [(ti, h) for ti in range(len(traces)) for h in range(24)
        if f"{ti}_{h}" not in done]
print(f"{len(jobs)} calls to make", flush=True)


def work(job):
    ti, h = job
    ms = matching(traces[ti], f"{DATE}T{h:02d}:00")
    return f"{ti}_{h}", (collapse(ms) if ms else [])


def save():
    json.dump({"date": DATE, "n_routes": len(traces), "links": done},
              open(OUT, "w"))


cnt = fails = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(work, j): j for j in jobs}
    for fut in as_completed(futs):
        key, links = fut.result()
        done[key] = links
        if not links:
            fails += 1
        cnt += 1
        if cnt % 50 == 0:
            save()
            nlinks = sum(len(v) for v in done.values())
            print(f"  {cnt}/{len(jobs)} calls, {nlinks} speed points, "
                  f"{fails} empty", flush=True)
save()
nlinks = sum(len(v) for v in done.values())
print(f"done: {cnt} calls, {nlinks} located speed points, {fails} empty",
      flush=True)
print(f"saved {OUT}")
