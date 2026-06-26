"""Fetch the arterial+ OSM graph for a city in data/cities.json.

Same network definition as the LA/NYC graphs: motorway..tertiary (+ links),
with osmnx free-flow speeds attached. Writes data/<city>_full.graphml.

Usage: fetch_city_graph.py <city>
"""
import json
import pathlib
import sys
import osmnx as ox

ROOT = pathlib.Path(__file__).resolve().parent.parent
CITY = sys.argv[1]
FILT = ('["highway"~"motorway|motorway_link|trunk|trunk_link|primary|'
        'primary_link|secondary|secondary_link|tertiary|tertiary_link"]')

cities = {c["name"]: c for c in json.load(open(ROOT / "data" / "cities.json"))}
if CITY not in cities:
    sys.exit(f"{CITY} not in cities.json ({list(cities)})")
w, s, e, n = cities[CITY]["bbox"]
out = ROOT / "data" / f"{CITY}_full.graphml"
print(f"fetching {CITY} arterial+ graph, bbox=({w},{s},{e},{n}) ...", flush=True)
G = ox.graph_from_bbox(bbox=(w, s, e, n), network_type="drive",
                       custom_filter=FILT, truncate_by_edge=True)
G = ox.add_edge_speeds(G)
ox.save_graphml(G, out)
print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {out.name}")
