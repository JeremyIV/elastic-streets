"""Breathing city from a sparse anchor-grid API pull (see anchor_pull.py).

Per hour: symmetrize anchor-edge times, fit the per-leg endpoint offset c
against the direct validation pairs, Dijkstra the anchor graph for dense
all-pairs times, then solve the same travel-time MDS as animate_mds.py
(one pooled global scale, warm starts x 2 passes). The full street fabric
rides along by inverse-distance interpolation of anchor displacements;
street colors come from the nearest anchor corridor's speed profile.

Times are Mapbox depart_at typical traffic, so the layout is intrinsically
calibrated: compose with TIME_SCALE 1.0 and no amplify step.

Usage: anchor_layout.py [pull_json] [graphml] [out_json]
"""

import json
import pathlib
import sys
import time

import numpy as np
import osmnx as ox
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
PULL = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "la_pull.json")
GRAPH = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "data" / "la_full.graphml")
OUT = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "data" / "day_mds_la.json")
ANCHOR_W = 0.05

pull = json.load(open(PULL))
K = len(pull["anchor_nodes"])
axy = np.array(pull["anchors_xy"])
edges = [tuple(e) for e in pull["edges"]]
val_pairs = [tuple(p) for p in pull["val_pairs"]]
E = len(edges)
print(f"{K} anchors, {E} edges")

# ---- per-hour symmetric edge times, gaps filled from neighboring hours ----
ew = np.full((E, 24), np.nan)
for h in range(24):
    tm = pull["times"][f"{h:02d}"]
    for k, (i, j) in enumerate(edges):
        ts = [tm.get(f"{i}-{j}"), tm.get(f"{j}-{i}")]
        ts = [t for t in ts if t]
        if ts:
            ew[k, h] = np.mean(ts)
miss = ~np.isfinite(ew)
if miss.any():
    for k in np.flatnonzero(miss.any(1)):
        good = np.flatnonzero(np.isfinite(ew[k]))
        if len(good):
            for h in np.flatnonzero(~np.isfinite(ew[k])):
                ew[k, h] = ew[k, good[np.argmin(np.minimum(
                    np.abs(good - h), 24 - np.abs(good - h)))]]
keep = np.isfinite(ew).all(1)
edges = [e for e, kp in zip(edges, keep) if kp]
ew = ew[keep]
E = len(edges)
print(f"{E} edges with full profiles ({(~keep).sum()} dropped)")

ii = np.array([e[0] for e in edges])
jj = np.array([e[1] for e in edges])

# ---- offset fit + dense all-pairs per hour ----
def allpairs(h, c):
    w = np.maximum(ew[:, h] - c, 30.0)
    adj = coo_matrix((np.r_[w, w], (np.r_[ii, jj], np.r_[jj, ii])),
                     shape=(K, K)).tocsr()
    return dijkstra(adj) + c

T_hours, offsets = [], []
for h in range(24):
    tm = pull["times"][f"{h:02d}"]
    direct = np.array([tm.get(f"V{i}-{j}") or np.nan for (i, j) in val_pairs])
    best = None
    for c in range(0, 301, 10):
        D = allpairs(h, c)
        ps = np.array([D[i, j] for (i, j) in val_pairs])
        err = np.nanmedian(np.abs(ps - direct) / direct)
        if best is None or err < best[1]:
            best = (c, err, D)
    c, err, D = best
    np.fill_diagonal(D, 0.0)
    T_hours.append(D)
    offsets.append(c)
    print(f"h={h:02d} offset {c:3d}s  val median |err| {err*100:4.1f}%")

# ---- MDS over anchors: same machinery as animate_mds, dense pairs ----
center = axy.mean(0)
P0 = axy - center
A0, B0 = np.triu_indices(K, 1)

num = den = 0.0
L0 = np.sqrt(((P0[A0] - P0[B0]) ** 2).sum(1))
for T in T_hours:
    Tij = T[A0, B0]
    kp = np.isfinite(Tij) & (Tij > 30)
    num += (L0[kp] * Tij[kp]).sum()
    den += (Tij[kp] ** 2).sum()
c_scale = num / den
print(f"global scale {c_scale:.2f} m/s ({c_scale*3.6:.0f} kph equivalent)")

span2 = ((P0.max(0) - P0.min(0)) ** 2).sum()
P0_ss = (P0 * P0).sum()

def solve_hour(T, X0):
    Tij = T[A0, B0]
    kp = np.isfinite(Tij) & (Tij > 30)
    A, B, D = A0[kp], B0[kp], c_scale * Tij[kp]
    wa = ANCHOR_W * len(A) / K / span2

    def f(x):
        P = x.reshape(K, 2)
        grad = np.zeros_like(P)
        dp = P[A] - P[B]
        L = np.sqrt((dp * dp).sum(1) + 1e-9)
        r = (L - D) / D
        Ev = (r * r).sum()
        g = (2.0 * r / (D * L))[:, None] * dp
        grad[:, 0] += np.bincount(A, g[:, 0], minlength=K)
        grad[:, 0] -= np.bincount(B, g[:, 0], minlength=K)
        grad[:, 1] += np.bincount(A, g[:, 1], minlength=K)
        grad[:, 1] -= np.bincount(B, g[:, 1], minlength=K)
        tmean = P.mean(0)
        Pc = P - tmean
        s = (Pc * P0).sum() / P0_ss
        dxy = Pc - s * P0
        Ev += wa * (dxy * dxy).sum()
        grad += 2.0 * wa * dxy
        return Ev, grad.ravel()

    res = minimize(f, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "maxfun": 4000})
    return res.x.reshape(K, 2), res.fun / len(A)

hours = []
X = P0
t0 = time.time()
for p in range(2):
    out = []
    for h in range(24):
        X, stress = solve_hour(T_hours[h], X)
        out.append(X)
        if p == 1:
            mean_r = np.sqrt((X ** 2).sum(1)).mean()
            r0 = np.sqrt((P0 ** 2).sum(1)).mean()
            print(f"  h={h:02d} stress/pair {stress:.4f} "
                  f"breath {mean_r/r0-1:+.1%} ({time.time()-t0:.0f}s)")
    hours = out

# ---- carry the street fabric: IDW of anchor displacements ----
print("loading street graph...")
G = ox.load_graphml(GRAPH)
G = ox.project_graph(G)
comp = max(__import__("networkx").weakly_connected_components(G), key=len)
G = G.subgraph(comp).copy()
nodes = list(G.nodes)
idx = {n: i for i, n in enumerate(nodes)}
Pg = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes]) - center
N = len(nodes)
print(f"{N} street nodes")

tree = cKDTree(P0)
d, nb = tree.query(Pg, k=8)
w = 1.0 / (d ** 2 + 200.0 ** 2)
w /= w.sum(1, keepdims=True)

node_hours = []
for X in hours:
    disp = X - P0
    node_hours.append(Pg + (w[..., None] * disp[nb]).sum(1))

# ---- street edges: geometry + nearest-corridor speed profile ----
elen = np.sqrt(((axy[ii] - axy[jj]) ** 2).sum(1))
v_corr = elen[:, None] / np.maximum(ew - np.array(offsets)[None, :], 30.0)
emid = (P0[ii] + P0[jj]) / 2
etree = cKDTree(emid)

edges_out = []
for u, v, k, dd in G.edges(keys=True, data=True):
    if "geometry" in dd:
        pts = np.array(dd["geometry"].coords) - center
    else:
        pts = np.array([Pg[idx[u]], Pg[idx[v]]])
    _, ce = etree.query(pts.mean(0))
    sp = v_corr[ce] * 3.6
    edges_out.append({
        "u": idx[u], "v": idx[v], "ff": round(float(sp.max()), 1),
        "sp": [round(float(s), 1) for s in sp],
        "obs": [True] * 24,
        "pts": np.round(pts, 1).ravel().tolist(),
    })

# ---- yardstick: layout distance per 10 true minutes, per hour ----
yard10 = []
for h, X in enumerate(hours):
    Tij = T_hours[h][A0, B0]
    m = (Tij >= 540) & (Tij <= 660)
    yard10.append(float(np.median(
        np.sqrt(((X[A0[m]] - X[B0[m]]) ** 2).sum(1)))))

out = {
    "meta": {
        "mode": "day-mds-anchor", "scale_mps": float(c_scale),
        "n_nodes": N, "n_anchors": K, "offsets_s": offsets,
        "yard10": yard10,
        "extent": [float(Pg[:, 0].min()), float(Pg[:, 0].max()),
                   float(Pg[:, 1].min()), float(Pg[:, 1].max())],
    },
    "nodes_flat": np.round(Pg, 1).ravel().tolist(),
    "node_hours": [np.round(Xn, 1).ravel().tolist() for Xn in node_hours],
    "edges": edges_out,
}
json.dump(out, open(OUT, "w"))
print(f"wrote {OUT} ({pathlib.Path(OUT).stat().st_size/1e6:.1f} MB)")
