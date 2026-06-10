"""Calibrate the 20-minute yardstick against the actual layouts.

For each hour: median straight-line layout distance between landmark pairs
whose congestion-aware network travel time is 18-22 min. Written into
day_mds.json meta as yard20 (24 values, meters of layout space).
"""

import json
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from animate_mds import load_pruned
from animate_mds import edge_hour_speeds, impute_speeds, DATA

GRAPH = sys.argv[2] if len(sys.argv) > 2 else str(DATA / "manhattan.graphml")
DATA_DIR = sys.argv[3] if len(sys.argv) > 3 else str(DATA)
G = load_pruned(GRAPH)
nodes = list(G.nodes)
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

ea, eb, length, ff = [], [], [], []
for u, v, k, d in G.edges(keys=True, data=True):
    ea.append(idx[u]); eb.append(idx[v])
    length.append(float(d["length"]))
    ff.append(float(d.get("speed_kph", 40.0)))
ea = np.array(ea); eb = np.array(eb)
length = np.array(length); ff = np.array(ff)
P0xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])

sp = np.clip(edge_hour_speeds(G, month=1, data_dir=DATA_DIR), 3.0, 110.0)
filled = impute_speeds(sp, ff, (P0xy[ea] + P0xy[eb]) / 2.0)

DAY_FILE = sys.argv[1] if len(sys.argv) > 1 else str(DATA / "day_mds.json")
day = json.load(open(DAY_FILE))
assert day["meta"]["n_nodes"] == N
hours = [np.array(h).reshape(N, 2) for h in day["node_hours"]]

rng = np.random.default_rng(11)
lm = rng.choice(N, 120, replace=False)
yard20, yard10 = [], []
for h in range(24):
    tt = length / (filled[:, h] / 3.6)
    adj = coo_matrix((np.r_[tt, tt], (np.r_[ea, eb], np.r_[eb, ea])),
                     shape=(N, N)).tocsr()
    T = dijkstra(adj, indices=lm)
    P = hours[h]
    D = np.sqrt(((P[lm][:, None, :] - P[None, :, :]) ** 2).sum(-1))
    ok = np.isfinite(T)
    m20 = ok & (T >= 1080) & (T <= 1320)
    m10 = ok & (T >= 540) & (T <= 660)
    yard20.append(float(np.median(D[m20])))
    yard10.append(float(np.median(D[m10])))
    print(f"h={h:02d}  20min ~ {yard20[-1]/1000:.2f} km   "
          f"10min ~ {yard10[-1]/1000:.2f} km")

day["meta"]["yard20"] = [round(y, 1) for y in yard20]
day["meta"]["yard10"] = [round(y, 1) for y in yard10]
json.dump(day, open(DAY_FILE, "w"))
print("old bar (c*1200):", round(day['meta']['scale_mps'] * 1200), "m")
print("saved yard20")
