"""Compare this graph's travel times against the public OSRM router.

OSRM is traffic-blind (profile speeds ~ typical uncongested driving), so the
fair comparison is against our overnight hours; our rush-hour times should be
substantially LONGER than OSRM.
"""

import json
import time
import urllib.request

import numpy as np
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from animate_mds import load_pruned
from animate_mds import edge_hour_speeds, impute_speeds

PLACES = {
    "Downtown":      (47.6105, -122.3370),
    "Northgate":     (47.7067, -122.3250),
    "Ballard":       (47.6687, -122.3849),
    "Capitol Hill":  (47.6195, -122.3210),
    "W Seattle Jct": (47.5612, -122.3871),
    "UW":            (47.6615, -122.3130),
    "Rainier Beach": (47.5223, -122.2670),
    "Magnolia":      (47.6400, -122.3990),
}
ROUTES = [
    ("Northgate", "Downtown"),
    ("Ballard", "Capitol Hill"),
    ("W Seattle Jct", "Downtown"),
    ("UW", "Downtown"),
    ("Rainier Beach", "Downtown"),
    ("Ballard", "W Seattle Jct"),
    ("Magnolia", "UW"),
    ("Northgate", "Rainier Beach"),
    ("Capitol Hill", "UW"),
]

print("loading graph + speeds...")
G = load_pruned("../data/seattle.graphml")
nodes = list(G.nodes)
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)
P = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])

ea, eb, length, ff = [], [], [], []
for u, v, k, d in G.edges(keys=True, data=True):
    ea.append(idx[u]); eb.append(idx[v])
    length.append(float(d["length"]))
    ff.append(float(d.get("speed_kph", 40.0)))
ea = np.array(ea); eb = np.array(eb)
length = np.array(length); ff = np.array(ff)
sp = np.clip(edge_hour_speeds(G, month=1, data_dir="../data/seattle"), 3, 110)
filled = impute_speeds(sp, ff, (P[ea] + P[eb]) / 2)

tf = Transformer.from_crs("EPSG:4326", G.graph["crs"], always_xy=True)
tree = cKDTree(P)
loc_idx = {}
for name, (lat, lon) in PLACES.items():
    x, y = tf.transform(lon, lat)
    _, j = tree.query([x, y])
    loc_idx[name] = j

def our_times(h):
    tt = length / (filled[:, h] / 3.6)
    adj = coo_matrix((np.r_[tt, tt], (np.r_[ea, eb], np.r_[eb, ea])),
                     shape=(N, N)).tocsr()
    srcs = sorted({loc_idx[a] for a, _ in ROUTES})
    T = dijkstra(adj, indices=srcs)
    row = {s: i for i, s in enumerate(srcs)}
    return {(a, b): T[row[loc_idx[a]], loc_idx[b]] / 60 for a, b in ROUTES}

t04 = our_times(4)
t17 = our_times(17)

def osrm(a, b):
    la, lo = PLACES[a]; lb, lob = PLACES[b]
    url = (f"https://router.project-osrm.org/route/v1/driving/"
           f"{lo},{la};{lob},{lb}?overview=false")
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)["routes"][0]["duration"] / 60

print(f"\n{'route':28s} {'OSRM':>6s} {'ours 4am':>9s} {'ours 5pm':>9s}")
r4, r17 = [], []
for a, b in ROUTES:
    o = osrm(a, b)
    time.sleep(1)
    print(f"{a+' -> '+b:28s} {o:5.0f}m {t04[(a,b)]:8.0f}m {t17[(a,b)]:8.0f}m")
    r4.append(t04[(a, b)] / o)
    r17.append(t17[(a, b)] / o)
print(f"\nmedian ours/OSRM: 4am {np.median(r4):.2f}x, 5pm {np.median(r17):.2f}x")
