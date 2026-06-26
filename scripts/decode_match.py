"""Decode collected Mapbox traffic tiles and stamp a congestion category onto
every arterial-graph edge, per hourly snapshot.

Precompute edge sample points once (projected). Per snapshot: decode all tiles
-> traffic segments (midpoint, class-group, severity) -> cKDTree -> for each
edge sample, nearest same-group segment within THRESH -> per-edge modal severity.
UTC snapshot stamp -> local hour via manifest tz. Output: per-edge x 24-hour
congestion-category matrix.

Usage: decode_match.py city graph.graphml [one|all]
  one (default): process first snapshot, print match-rate diagnostics
  all: process every snapshot, write data/<city>_edge_cong.json
"""
import gzip
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import osmnx as ox
import pyproj
from scipy.spatial import cKDTree
from shapely.geometry import LineString
import mapbox_vector_tile as mvt

ROOT = pathlib.Path(__file__).resolve().parent.parent
CITY = sys.argv[1]
GRAPH = pathlib.Path(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "one"
SNAPROOT = ROOT / "data" / "traffic_snaps" / CITY
Z, EXTENT, THRESH = 13, 4096, 35.0
SEV = {"low": 0, "moderate": 1, "heavy": 2, "severe": 3}
SEVNAME = {v: k for k, v in SEV.items()}
LIMITED = {"motorway", "motorway_link", "trunk", "trunk_link"}


def grp(cls):
    return 0 if cls in LIMITED else 1


def tile_bounds(x, y, z):
    n = 2 ** z
    lon0 = x / n * 360 - 180
    lon1 = (x + 1) / n * 360 - 180
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lon0, lat1, lon1, lat0  # w,s,e,n  (lat0=north since y down)


manifest = json.load(open(SNAPROOT / "manifest.json"))
TZ = ZoneInfo(manifest["tz"])
print(f"{CITY}: tz={manifest['tz']}", flush=True)

G = ox.load_graphml(GRAPH)
Gp = ox.project_graph(G)
CRS = Gp.graph["crs"]
to_utm = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

E = list(Gp.edges(keys=True))
print(f"{len(E)} edges; building sample points ...", flush=True)


def edge_geom(u, v, k):
    d = Gp.edges[u, v, k]
    if "geometry" in d:
        return d["geometry"]
    return LineString([(Gp.nodes[u]["x"], Gp.nodes[u]["y"]),
                       (Gp.nodes[v]["x"], Gp.nodes[v]["y"])])


samp_x, samp_y, samp_eidx = [], [], []
edge_grp = np.empty(len(E), dtype=int)
edge_cls = []
for ei, (u, v, k) in enumerate(E):
    cls = Gp.edges[u, v, k].get("highway", "?")
    cls = cls[0] if isinstance(cls, list) else cls
    edge_cls.append(cls)
    edge_grp[ei] = grp(cls)
    geom = edge_geom(u, v, k)
    L = geom.length
    npt = max(2, int(L / 25) + 1)
    for s in np.linspace(0, L, npt):
        p = geom.interpolate(s)
        samp_x.append(p.x); samp_y.append(p.y); samp_eidx.append(ei)
samp_x = np.array(samp_x); samp_y = np.array(samp_y)
samp_eidx = np.array(samp_eidx)
samp_grp = edge_grp[samp_eidx]
samp_xy = np.column_stack([samp_x, samp_y])
print(f"{len(samp_x)} edge sample points", flush=True)


def decode_snapshot(sdir):
    """All tiles in a snapshot -> arrays of traffic segment midpoints."""
    mx, my, mg, ms = [], [], [], []
    for t in sdir.glob("*.mvt"):
        tx, ty = map(int, t.stem.split("_"))
        w, s, e, n = tile_bounds(tx, ty, Z)
        raw = t.read_bytes()
        try:
            data = mvt.decode(gzip.decompress(raw))
        except Exception:
            try:
                data = mvt.decode(raw)
            except Exception:
                continue
        for f in data.get("traffic", {}).get("features", []):
            pr = f["properties"]
            sev = SEV.get(pr.get("congestion"))
            if sev is None:
                continue
            g = grp(pr.get("class", ""))
            geom = f["geometry"]
            lines = (geom["coordinates"] if geom["type"] == "MultiLineString"
                     else [geom["coordinates"]])
            for line in lines:
                for a, b in zip(line[:-1], line[1:]):
                    # tile-local -> lonlat (y up = north, EXTENT scale)
                    lon = w + ((a[0] + b[0]) / 2 / EXTENT) * (e - w)
                    lat = s + ((a[1] + b[1]) / 2 / EXTENT) * (n - s)
                    mx.append(lon); my.append(lat); mg.append(g); ms.append(sev)
    X, Y = to_utm.transform(mx, my)
    return np.array(X), np.array(Y), np.array(mg), np.array(ms)


def match(sdir):
    X, Y, mg, ms = decode_snapshot(sdir)
    if len(X) == 0:
        return np.full(len(E), -1)
    # separate trees per group so nearest respects group
    per_edge_votes = defaultdict(list)
    for g in (0, 1):
        seg_mask = mg == g
        if not seg_mask.any():
            continue
        tree = cKDTree(np.column_stack([X[seg_mask], Y[seg_mask]]))
        sev_g = ms[seg_mask]
        smp_mask = samp_grp == g
        if not smp_mask.any():
            continue
        dist, idx = tree.query(samp_xy[smp_mask], k=1,
                               distance_upper_bound=THRESH)
        ok = np.isfinite(dist)
        eidx_g = samp_eidx[smp_mask]
        for ei, si in zip(eidx_g[ok], idx[ok]):
            per_edge_votes[ei].append(int(sev_g[si]))
    cats = np.full(len(E), -1)
    for ei, votes in per_edge_votes.items():
        c = Counter(votes)
        top = max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]  # mode,tie->severe
        cats[ei] = top
    return cats


snaps = sorted(SNAPROOT.glob("snap_*"))
print(f"{len(snaps)} snapshots", flush=True)

if MODE == "one":
    cats = match(snaps[0])
    matched = (cats >= 0).sum()
    print(f"\n=== {snaps[0].name} ===")
    print(f"matched {matched}/{len(E)} edges = {100*matched/len(E):.1f}%")
    dist = Counter(SEVNAME.get(int(c), "none") for c in cats)
    print("edge category distribution:", dict(dist))
    # match rate by class group
    for g, gname in [(0, "limited"), (1, "arterial")]:
        m = edge_grp == g
        mm = (cats[m] >= 0).sum()
        print(f"  {gname:9s}: {mm}/{m.sum()} = {100*mm/max(m.sum(),1):.1f}%")
    print("\nif match rate is low (<70%), y-axis may be flipped — investigate")
else:
    def local_hour(stamp):
        dt = datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
        return dt.astimezone(TZ).hour
    by_hour = defaultdict(list)  # hour -> list of cats arrays
    for si, sdir in enumerate(snaps):
        stamp = sdir.name.replace("snap_", "")
        hr = local_hour(stamp)
        cats = match(sdir)
        by_hour[hr].append(cats)
        print(f"  {sdir.name} -> local hr {hr:02d}, "
              f"matched {100*(cats>=0).mean():.0f}%", flush=True)
    # 24-hour matrix: combine duplicate hours by rounded-mean severity
    hourly = {}
    for hr in range(24):
        if hr in by_hour:
            stack = np.array(by_hour[hr])     # (dups, nedges)
            valid = stack >= 0
            cnt = valid.sum(axis=0)
            sm = np.where(valid, stack, 0).sum(axis=0)
            mean = sm / np.maximum(cnt, 1)
            hourly[hr] = np.where(cnt > 0, np.rint(mean).astype(int), -1).tolist()
    out = {"city": CITY, "edge_order": [[int(u), int(v), int(k)] for u, v, k in E],
           "edge_class": edge_cls, "hours": sorted(hourly),
           "cong": {str(h): hourly[h] for h in hourly}}
    json.dump(out, open(ROOT / "data" / f"{CITY}_edge_cong.json", "w"))
    covered = sorted(hourly)
    print(f"\n{len(covered)}/24 local hours covered: {covered}")
    print(f"saved data/{CITY}_edge_cong.json")
