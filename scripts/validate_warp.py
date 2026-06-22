"""Validate a warped layout's distances against the Mapbox Directions API.

For N random node pairs at random hours: compare the layout's "travel-time-space"
distance (||layout_i - layout_j|| / scale_c, in seconds) to the actual driving
time the Directions API returns for those lon/lats at that depart_at hour. Saves
a scatter PNG. Tests whether the map's distances -- and its rush-hour expansion
-- match reality. Compares the canonical (dir) and sampled layouts on shared
API calls. Needs .mapbox_token.

Usage: validate_warp.py [n_pairs]
"""
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pyproj

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
DATE = "2026-06-22"                                   # a future Monday (same weekday the maps used)
NPAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 50

RAMP = {"motorway_link", "trunk_link", "primary_link", "secondary_link"}


def cls_of(d):
    h = d.get("highway", "?")
    return h[0] if isinstance(h, list) else h


def prune(Gu):                                        # identical to solve_directions
    while True:
        drop = [n for n in Gu.nodes if Gu.degree(n) == 1 and
                cls_of(next(iter(Gu.edges(n, data=True)))[2]) in RAMP]
        if not drop:
            break
        Gu.remove_nodes_from(drop)
    return Gu.subgraph(max(nx.connected_components(Gu), key=len)).copy()


# reproduce solve_directions node ordering, get lon/lat per node index
G = ox.load_graphml(DATA / "manhattan.graphml")
crs = G.graph.get("crs", "epsg:4326")
sample_x = float(next(iter(G.nodes(data=True)))[1]["x"])
Gp = G if abs(sample_x) > 1000 else ox.project_graph(G)
Gu = prune(nx.MultiGraph(ox.convert.to_undirected(Gp)))
nodes = list(Gu.nodes)
N = len(nodes)
to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
xs = np.array([float(Gu.nodes[n]["x"]) for n in nodes])
ys = np.array([float(Gu.nodes[n]["y"]) for n in nodes])
lon, lat = to_ll.transform(xs, ys)

# verify index alignment against the layout's projected+centered nodes_flat
ref = json.load(open(DATA / "day_mds_manhattan_dir.json"))
P0 = np.array(ref["nodes_flat"]).reshape(-1, 2)
Pp = np.array([[float(Gu.nodes[n]["x"]), float(Gu.nodes[n]["y"])] for n in nodes])
Pp -= Pp.mean(0)
err = np.hypot(*(Pp - P0).T).max()
print(f"{N} nodes; alignment err {err:.1f} m; lon {lon.min():.2f}..{lon.max():.2f}, "
      f"lat {lat.min():.2f}..{lat.max():.2f}", flush=True)

rng = np.random.default_rng(0)
ii = rng.integers(0, N, NPAIRS); jj = rng.integers(0, N, NPAIRS)
hh = rng.integers(0, 24, NPAIRS)
ok = ii != jj
ii, jj, hh = ii[ok], jj[ok], hh[ok]


def api_tt(i, j, h):
    coords = f"{lon[i]},{lat[i]};{lon[j]},{lat[j]}"
    params = {"access_token": TOKEN, "overview": "false",
              "depart_at": f"{DATE}T{h:02d}:00"}
    url = (f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
           f"{urllib.parse.quote(coords)}?{urllib.parse.urlencode(params)}")
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                d = json.loads(r.read().decode())
            if d.get("code") == "Ok" and d.get("routes"):
                return d["routes"][0]["duration"]
            return None
        except Exception:
            time.sleep(2)
    return None


api = np.full(len(ii), np.nan)
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(api_tt, int(ii[k]), int(jj[k]), int(hh[k])): k
            for k in range(len(ii))}
    for f in as_completed(futs):
        d = f.result()
        if d:
            api[futs[f]] = d
print(f"got {int(np.isfinite(api).sum())}/{len(ii)} API travel times", flush=True)


def map_tt(path):
    d = json.load(open(path)); c = d["meta"]["scale_mps"]
    nh = [np.array(x).reshape(-1, 2) for x in d["node_hours"]]
    out = np.full(len(ii), np.nan)
    for k in range(len(ii)):
        P = nh[hh[k]]
        out[k] = np.hypot(*(P[ii[k]] - P[jj[k]])) / c
    return out


mdir = map_tt(DATA / "day_mds_manhattan_dir.json")
msmp = map_tt(DATA / "day_mds_manhattan_sampled.json")

g = np.isfinite(api)
RUSH = {7, 8, 9, 16, 17, 18}
rush = np.array([h in RUSH for h in hh])
for m, name in [(mdir, "canonical"), (msmp, "sampled")]:
    sl = np.polyfit(m[g], api[g], 1)[0]
    gr, go = g & rush, g & ~rush
    print(f"{name:10s}: slope(api~map) {sl:.2f}  median api/map {np.median(api[g]/m[g]):.2f}"
          f"  | rush api/map {np.median(api[gr]/m[gr]):.2f}  offpeak {np.median(api[go]/m[go]):.2f}",
          flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
for ax, m, name in [(axes[0], mdir, "canonical  1/D^2 landmark"),
                    (axes[1], msmp, "sampled  2-hop ew=100")]:
    sc = ax.scatter(m[g] / 60, api[g] / 60, c=hh[g], cmap="twilight",
                    s=45, vmin=0, vmax=24, edgecolor="k", linewidth=0.3)
    lim = max(np.nanmax(m[g]), np.nanmax(api[g])) / 60 * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x")
    sl, b = np.polyfit(m[g] / 60, api[g] / 60, 1)
    xs = np.array([0, lim])
    ax.plot(xs, sl * xs + b, "r-", lw=1.3, alpha=0.8, label=f"fit (slope {sl:.2f})")
    ax.set_xlabel("map distance / c   (min, 'travel-time space')")
    ax.set_ylabel("Mapbox API travel time (min)")
    ax.set_title(name); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal"); ax.legend(loc="lower right", fontsize=8)
fig.colorbar(sc, ax=axes, label="hour of day (rush ~7-9, 16-18)", fraction=0.025)
fig.suptitle(f"Warp distance vs real Mapbox travel time  ({int(g.sum())} pairs, "
             f"random hours)")
out = ROOT / "shots" / "validate_warp.png"
fig.savefig(out, dpi=115, bbox_inches="tight")
print(f"saved {out}")
