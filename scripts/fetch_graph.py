"""Fetch Manhattan's drivable street network and tag each edge with a free-flow speed.

Speeds come from OSM maxspeed tags where present; otherwise fall back to
per-highway-class free-flow estimates (kph) tuned for NYC.
"""

import pathlib

import osmnx as ox

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

# Free-flow speeds in kph by OSM highway class, used when maxspeed is missing.
# NYC default limit is 25 mph (~40 kph) on local streets; highways run faster.
HWY_SPEEDS = {
    "motorway": 90,
    "motorway_link": 60,
    "trunk": 75,
    "trunk_link": 50,
    "primary": 50,
    "primary_link": 40,
    "secondary": 45,
    "secondary_link": 35,
    "tertiary": 40,
    "tertiary_link": 32,
    "residential": 30,
    "living_street": 15,
    "unclassified": 32,
}

# "Manhattan Island" (not the borough) keeps Randall's, Roosevelt etc. out
import sys
PLACE = sys.argv[1] if len(sys.argv) > 1 else "Manhattan Island, New York, USA"
OUT = sys.argv[2] if len(sys.argv) > 2 else "manhattan.graphml"
print(f"Downloading drive network for {PLACE}...")
G = ox.graph_from_place(PLACE, network_type="drive", simplify=True)
print(f"Raw graph: {len(G.nodes)} nodes, {len(G.edges)} edges")

G = ox.routing.add_edge_speeds(G, hwy_speeds=HWY_SPEEDS)
G = ox.project_graph(G)  # UTM, meters

out = DATA / OUT
ox.save_graphml(G, out)
print(f"Saved {out}")
