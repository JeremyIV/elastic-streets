"""Pilot part 3: attach OSM road class to each probe link via the city graph.

Fetches (and caches) the NYC arterial+ graph, then for every link from
pilot_links.py finds the nearest graph edge and reads its highway class.
Links whose nearest edge is >40 m away aren't on a graphed road -> dropped
(same filter the real calibration will use). Output: the joined training rows
(class, category, numeric, speed) that the full calibration harvests at scale.
"""
import json
import pathlib
import osmnx as ox

ROOT = pathlib.Path(__file__).resolve().parent.parent
NYC_BBOX = (-74.05, 40.54, -73.70, 40.92)  # from data/cities.json
FILT = ('["highway"~"motorway|motorway_link|trunk|trunk_link|primary|'
        'primary_link|secondary|secondary_link|tertiary|tertiary_link"]')
gpath = ROOT / "data" / "nyc_full.graphml"

if gpath.exists():
    print("loading cached", gpath.name)
    G = ox.load_graphml(gpath)
else:
    print("fetching NYC arterial+ graph (bbox", NYC_BBOX, ")...", flush=True)
    G = ox.graph_from_bbox(bbox=NYC_BBOX, network_type="drive",
                           custom_filter=FILT, truncate_by_edge=True)
    G = ox.add_edge_speeds(G)
    ox.save_graphml(G, gpath)
    print("saved", gpath.name)
print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

Gp = ox.project_graph(G)
links = json.loads((ROOT / "data" / "pilot_nyc_links.json").read_text())
lons = [l["lon"] for l in links]
lats = [l["lat"] for l in links]
# project points to the graph's CRS
import pyproj
to_utm = pyproj.Transformer.from_crs("EPSG:4326", Gp.graph["crs"], always_xy=True)
xs, ys = to_utm.transform(lons, lats)
edges, dists = ox.distance.nearest_edges(Gp, xs, ys, return_dist=True)

def cls(u, v, k):
    h = Gp.edges[u, v, k].get("highway", "?")
    return h[0] if isinstance(h, list) else h

from collections import Counter
kept, classcount = [], Counter()
for l, (u, v, k), d in zip(links, edges, dists):
    if d > 40:  # not on a graphed (arterial+) road -> drop
        continue
    c = cls(u, v, k)
    l["class"], l["match_m"] = c, round(d, 1)
    kept.append(l)
    classcount[c] += 1

print(f"\n{len(kept)}/{len(links)} links matched within 40 m")
print("class distribution:", dict(classcount))
print(f"\n{'class':14s} {'cat':9s} {'num':>4s} {'len_m':>6s} {'kmh':>5s} "
      f"{'maxsp':>5s} {'match_m':>7s}")
for l in kept[:14] + kept[-6:]:
    print(f"{l['class']:14s} {l['cat']:9s} {str(l['numeric']):>4s} "
          f"{l['len_m']:6.0f} {l['kmh_distdur']:5.1f} {str(l['maxspeed']):>5s} "
          f"{l['match_m']:7.1f}")

# the joined rows ARE calibration training data: (class, category) -> speed
print("\n=== speed by (class, category), the calibration target ===")
import statistics
cells = {}
for l in kept:
    cells.setdefault((l["class"], l["cat"]), []).append(l["kmh_distdur"])
for (c, cat), v in sorted(cells.items()):
    print(f"  {c:14s} {cat:9s} n={len(v):3d}  median {statistics.median(v):5.1f} kmh")
