"""Validate ANY city's warped layout vs live Mapbox Directions travel times.

For N random node pairs at random hours: compare the layout's travel-time-space
distance (||layout_i - layout_j|| / scale_c, seconds) to the actual Mapbox
driving-traffic duration (depart_at that hour). Scatter colored by hour, with a
per-hour-bucket api/map ratio table so you can SEE whether rush hours are
systematically under-expanded (map underestimates congestion).

Usage: validate_city.py <city> [layout_json] [n_pairs]
  layout_json defaults to data/day_mds_<city>_gold.json
Needs .mapbox_token. Reproduces solve_directions' node ordering from
data/<city>_full.graphml, so the layout must come from that graph.
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
DATE = "2026-06-29"                                   # an upcoming Monday (future depart_at)

CITY = sys.argv[1]
LAYOUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else DATA / f"day_mds_{CITY}_gold.json"
NPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 100

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
G = ox.load_graphml(DATA / f"{CITY}_full.graphml")
sample_x = float(next(iter(G.nodes(data=True)))[1]["x"])
Gp = G if abs(sample_x) > 1000 else ox.project_graph(G)   # project to metric if stored as lon/lat
Gu = prune(nx.MultiGraph(ox.convert.to_undirected(Gp)))
nodes = list(Gu.nodes)
N = len(nodes)
# transform from the PROJECTED graph's actual CRS back to lon/lat (works whether
# the source graphml was already projected or we just projected it above)
to_ll = pyproj.Transformer.from_crs(Gp.graph["crs"], "EPSG:4326", always_xy=True)
xs = np.array([float(Gu.nodes[n]["x"]) for n in nodes])
ys = np.array([float(Gu.nodes[n]["y"]) for n in nodes])
lon, lat = to_ll.transform(xs, ys)

# verify index alignment against the layout's projected+centered nodes_flat
d = json.load(open(LAYOUT))
P0 = np.array(d["nodes_flat"]).reshape(-1, 2)
c = d["meta"]["scale_mps"]
nh = [np.array(x).reshape(-1, 2) for x in d["node_hours"]]
if len(P0) != N:
    sys.exit(f"ABORT: graph has {N} nodes but layout has {len(P0)} -- ordering mismatch")
Pp = np.array([[float(Gu.nodes[n]["x"]), float(Gu.nodes[n]["y"])] for n in nodes])
Pp -= Pp.mean(0)
err = np.hypot(*(Pp - P0).T).max()
print(f"{CITY}: {N} nodes; alignment err {err:.1f} m (should be ~0); "
      f"lon {lon.min():.2f}..{lon.max():.2f} lat {lat.min():.2f}..{lat.max():.2f}", flush=True)
if err > 50:
    sys.exit("ABORT: alignment error too large -- node ordering does not match the layout")

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
                dd = json.loads(r.read().decode())
            if dd.get("code") == "Ok" and dd.get("routes"):
                return dd["routes"][0]["duration"]
            return None
        except Exception:
            time.sleep(2)
    return None


api = np.full(len(ii), np.nan)
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(api_tt, int(ii[k]), int(jj[k]), int(hh[k])): k for k in range(len(ii))}
    for f in as_completed(futs):
        v = f.result()
        if v:
            api[futs[f]] = v
print(f"got {int(np.isfinite(api).sum())}/{len(ii)} API travel times", flush=True)

mp = np.array([np.hypot(*(nh[hh[k]][ii[k]] - nh[hh[k]][jj[k]])) / c for k in range(len(ii))])
g = np.isfinite(api)

# per-hour-bucket api/map ratio: is rush systematically under-expanded?
print("\n hour-bucket   n   median api/map  (ratio>1 => map UNDERestimates real time)")
buckets = [("night 0-5", range(0, 6)), ("amrush 6-9", range(6, 10)),
           ("midday 10-15", range(10, 16)), ("pmrush 16-19", range(16, 20)),
           ("evening 20-23", range(20, 24))]
for name, hrs in buckets:
    m = g & np.isin(hh, list(hrs))
    if m.sum():
        print(f"  {name:13s} {int(m.sum()):3d}   {np.median(api[m]/mp[m]):.2f}")
sl = np.polyfit(mp[g], api[g], 1)[0]
print(f"\noverall: slope(api~map) {sl:.2f}  median api/map {np.median(api[g]/mp[g]):.2f}")

fig, ax = plt.subplots(figsize=(7.5, 7))
sc = ax.scatter(mp[g] / 60, api[g] / 60, c=hh[g], cmap="twilight", s=55,
                vmin=0, vmax=24, edgecolor="k", linewidth=0.3)
lim = max(np.nanmax(mp[g]), np.nanmax(api[g])) / 60 * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x (perfect)")
b = np.polyfit(mp[g] / 60, api[g] / 60, 1)
ax.plot([0, lim], [b[1], b[0] * lim + b[1]], "r-", lw=1.3, label=f"fit (slope {b[0]:.2f})")
ax.set_xlabel("warped-map distance / c   (min)")
ax.set_ylabel("Mapbox API travel time (min)")
ax.set_title(f"{CITY}: warped map vs real Mapbox time ({int(g.sum())} pairs)")
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
ax.legend(loc="lower right")
fig.colorbar(sc, ax=ax, label="hour of day (rush ~7-9, 16-19)", fraction=0.046)
out = ROOT / "shots" / f"validate_{CITY}.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"saved {out}")
