"""Manhattan version of validate_times.py: graph times vs public OSRM."""

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
    "Inwood":          (40.8677, -73.9212),
    "Harlem 125th":    (40.8075, -73.9454),
    "Columbus Circle": (40.7681, -73.9819),
    "Midtown East":    (40.7549, -73.9707),
    "Chelsea":         (40.7465, -74.0014),
    "East Village":    (40.7265, -73.9815),
    "Wall St":         (40.7074, -74.0113),
    "UWS 86th":        (40.7870, -73.9754),
}
ROUTES = [
    ("Inwood", "Midtown East"),
    ("Harlem 125th", "Wall St"),
    ("Columbus Circle", "East Village"),
    ("UWS 86th", "Wall St"),
    ("Chelsea", "Harlem 125th"),
    ("Midtown East", "Wall St"),
    ("Inwood", "Wall St"),
    ("East Village", "UWS 86th"),
    ("Columbus Circle", "Wall St"),
]

print("loading graph + speeds...")
G = load_pruned("../data/manhattan.graphml")
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
sp = np.clip(edge_hour_speeds(G, month=1, data_dir="../data"), 3, 110)
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

print(f"\n{'route':32s} {'OSRM':>6s} {'ours 4am':>9s} {'ours 5pm':>9s}")
r4, r17 = [], []
for a, b in ROUTES:
    o = osrm(a, b)
    time.sleep(1)
    print(f"{a+' -> '+b:32s} {o:5.0f}m {t04[(a,b)]:8.0f}m {t17[(a,b)]:8.0f}m")
    r4.append(t04[(a, b)] / o)
    r17.append(t17[(a, b)] / o)
print(f"\nmedian ours/OSRM: 4am {np.median(r4):.2f}x, 5pm {np.median(r17):.2f}x")
