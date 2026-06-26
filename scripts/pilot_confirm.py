"""Pilot part 5: confirm the speed annotation gives a coherent (class, category)
-> speed table (low > moderate > heavy), vs the broken distance/duration basis.
Cached graph -> fast.
"""
import json
import pathlib
import statistics
import osmnx as ox
import pyproj

ROOT = pathlib.Path(__file__).resolve().parent.parent
G = ox.load_graphml(ROOT / "data" / "nyc_full.graphml")
Gp = ox.project_graph(G)
links = json.loads((ROOT / "data" / "pilot_nyc_links.json").read_text())
to_utm = pyproj.Transformer.from_crs("EPSG:4326", Gp.graph["crs"], always_xy=True)
xs, ys = to_utm.transform([l["lon"] for l in links], [l["lat"] for l in links])
edges, dists = ox.distance.nearest_edges(Gp, xs, ys, return_dist=True)

def cls(u, v, k):
    h = Gp.edges[u, v, k].get("highway", "?")
    return h[0] if isinstance(h, list) else h

CATORD = {"low": 0, "moderate": 1, "heavy": 2, "severe": 3, "unknown": 9}
cells = {}
for l, (u, v, k), d in zip(links, edges, dists):
    if d > 40:
        continue
    cells.setdefault((cls(u, v, k), l["cat"]), {"dd": [], "ann": []})
    cells[(cls(u, v, k), l["cat"])]["dd"].append(l["kmh_distdur"])
    cells[(cls(u, v, k), l["cat"])]["ann"].append(l["kmh_ann"])

print(f"{'class':14s} {'category':9s} {'n':>3s} {'distdur':>8s} {'ANNOT':>8s}")
last = None
for (c, cat), v in sorted(cells.items(), key=lambda x: (x[0][0], CATORD[x[0][1]])):
    if c != last:
        print("  " + "-" * 44)
        last = c
    print(f"{c:14s} {cat:9s} {len(v['ann']):3d} "
          f"{statistics.median(v['dd']):8.1f} {statistics.median(v['ann']):8.1f}")
print("\n(ANNOT column should fall low->moderate->heavy within each class)")
