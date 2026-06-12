"""Full-resolution breathing city from the sparse anchor pull: instead of
IDW-warping streets between 253 solved anchors (visibly low-rank), give
every street edge a corridor-derived speed and solve the same full-graph
landmark MDS as the Uber cities — every intersection individually placed.
Zero additional API calls; the anchor pull is the speed sensor.

Corridor matching is class-aware: motorway street edges only inherit from
freeway corridors, surface streets only from surface corridors (nearest-
anyone matching let streets beside a freeway move at freeway speed).

Usage: anchor_layout_full.py [pull_json] [graphml] [out_json] [landmarks]
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
OUT = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "data" / "day_mds_la_full.json")
N_LM = int(sys.argv[4]) if len(sys.argv) > 4 else 400
ANCHOR_W = 0.05

# ---- anchor corridor speeds (same offset machinery as anchor_layout) ----
pull = json.load(open(PULL))
K = len(pull["anchor_nodes"])
NFWY = pull["n_fwy_anchors"]
axy = np.array(pull["anchors_xy"])
a_edges = [tuple(e) for e in pull["edges"]]
val_pairs = [tuple(p) for p in pull["val_pairs"]]
E = len(a_edges)

ew = np.full((E, 24), np.nan)
for h in range(24):
    tm = pull["times"][f"{h:02d}"]
    for k, (i, j) in enumerate(a_edges):
        ts = [t for t in (tm.get(f"{i}-{j}"), tm.get(f"{j}-{i}")) if t]
        if ts:
            ew[k, h] = np.mean(ts)
for k in np.flatnonzero((~np.isfinite(ew)).any(1)):
    good = np.flatnonzero(np.isfinite(ew[k]))
    for h in np.flatnonzero(~np.isfinite(ew[k])):
        ew[k, h] = ew[k, good[np.argmin(np.minimum(
            np.abs(good - h), 24 - np.abs(good - h)))]]

ii = np.array([e[0] for e in a_edges])
jj = np.array([e[1] for e in a_edges])
offsets = []
for h in range(24):
    tm = pull["times"][f"{h:02d}"]
    direct = np.array([tm.get(f"V{i}-{j}") or np.nan for (i, j) in val_pairs])
    best = None
    for c in range(0, 301, 10):
        w = np.maximum(ew[:, h] - c, 30.0)
        adj = coo_matrix((np.r_[w, w], (np.r_[ii, jj], np.r_[jj, ii])),
                         shape=(K, K)).tocsr()
        D = dijkstra(adj, indices=sorted({p[0] for p in val_pairs}))
        row = {s: r for r, s in enumerate(sorted({p[0] for p in val_pairs}))}
        ps = np.array([D[row[i], j] + c for (i, j) in val_pairs])
        err = np.nanmedian(np.abs(ps - direct) / direct)
        if best is None or err < best[1]:
            best = (c, err)
    offsets.append(best[0])
offsets = np.array(offsets, float)
print(f"offsets fit: {offsets.astype(int).tolist()}")

is_fwy_corr = (ii < NFWY) & (jj < NFWY)
print(f"{E} corridors ({is_fwy_corr.sum()} freeway-class)")

# ---- street graph ----
print("loading street graph...")
G = ox.load_graphml(GRAPH)
G = ox.project_graph(G)
import networkx as nx
G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
nodes = list(G.nodes)
idx = {n: i for i, n in enumerate(nodes)}
center = axy.mean(0)
Pg = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes]) - center
N = len(nodes)

# corridor speeds against actual ROAD length between the anchor endpoints,
# not the straight-line chord — chord/(t-c) understates surface speeds by
# the grid detour factor (~1.3x) while freeways stay near 1.0x, which made
# the surface fabric carry too much area and buckle against the skeleton
anchor_sidx = np.array([idx.get(int(n), -1) for n in pull["anchor_nodes"]])
g_ea, g_eb, g_len = [], [], []
for u, v, k, d in G.edges(keys=True, data=True):
    g_ea.append(idx[u]); g_eb.append(idx[v]); g_len.append(float(d["length"]))
g_ea = np.array(g_ea); g_eb = np.array(g_eb); g_len = np.array(g_len)
adj_len = coo_matrix((np.r_[g_len, g_len], (np.r_[g_ea, g_eb], np.r_[g_eb, g_ea])),
                     shape=(N, N)).tocsr()
src = anchor_sidx[anchor_sidx >= 0]
Droad = dijkstra(adj_len, indices=src)
row = {s: r for r, s in enumerate(src)}
chord = np.sqrt(((axy[ii] - axy[jj]) ** 2).sum(1))
road = chord.copy()
for k in range(E):
    a, b = anchor_sidx[ii[k]], anchor_sidx[jj[k]]
    if a >= 0 and b >= 0 and np.isfinite(Droad[row[a], b]):
        road[k] = max(Droad[row[a], b], chord[k])
print(f"road/chord detour: median {np.median(road/chord):.2f} "
      f"(fwy {np.median((road/chord)[is_fwy_corr]):.2f}, "
      f"surface {np.median((road/chord)[~is_fwy_corr]):.2f})")
v_corr = road[:, None] / np.maximum(ew - offsets[None, :], 30.0) * 3.6  # kph
P0a = axy - center
emid = (P0a[ii] + P0a[jj]) / 2
tree_f = cKDTree(emid[is_fwy_corr])
tree_s = cKDTree(emid[~is_fwy_corr])
map_f = np.flatnonzero(is_fwy_corr)
map_s = np.flatnonzero(~is_fwy_corr)

# regional hourly congestion (median surface corridor, vs its own best hour)
v_surf = v_corr[~is_fwy_corr]
regional = np.median(v_surf / v_surf.max(1, keepdims=True), axis=0)

ea, eb, elen, geom, espd = [], [], [], [], []
for u, v, k, d in G.edges(keys=True, data=True):
    hw = d.get("highway")
    hw = hw if isinstance(hw, list) else [hw]
    fwy = any("motorway" in str(h) for h in hw)
    if "geometry" in d:
        pts = np.array(d["geometry"].coords) - center
    else:
        pts = np.array([Pg[idx[u]], Pg[idx[v]]])
    mid = pts.mean(0)
    # IDW over the 6 nearest same-class corridors: hard nearest-corridor
    # cells turn speed discontinuities into position discontinuities (fur)
    if fwy:
        dmin, ce = tree_f.query(mid, k=min(6, len(map_f)))
        ce = map_f[np.atleast_1d(ce)]
    else:
        dmin, ce = tree_s.query(mid, k=min(6, len(map_s)))
        ce = map_s[np.atleast_1d(ce)]
    wk = 1.0 / (np.atleast_1d(dmin) + 500.0)
    prof = (wk[:, None] * v_corr[ce]).sum(0) / wk.sum()
    # far from any corridor the match is unrepresentative: blend toward the
    # edge's own free-flow speed under the regional congestion curve
    wf = np.clip((float(np.atleast_1d(dmin)[0]) - 1500.0) / 3000.0, 0.0, 1.0)
    if wf > 0:
        ff = float(d.get("speed_kph", 40.0))
        prof = (1 - wf) * prof + wf * ff * regional
    ea.append(idx[u]); eb.append(idx[v])
    elen.append(float(d["length"]))
    geom.append(pts)
    espd.append(prof)
ea = np.array(ea); eb = np.array(eb)
elen = np.array(elen)
sp = np.clip(np.array(espd), 3.0, 110.0)          # (E_street, 24) kph
print(f"{N} nodes, {len(ea)} street edges with corridor speeds")

# ---- anchor-level MDS (trusted global shape), used to pin the full solve ----
P0a = axy - center
Aa, Ba = np.triu_indices(K, 1)
T_anch = []
for h in range(24):
    w = np.maximum(ew[:, h] - offsets[h], 30.0)
    adj = coo_matrix((np.r_[w, w], (np.r_[ii, jj], np.r_[jj, ii])),
                     shape=(K, K)).tocsr()
    D = dijkstra(adj) + offsets[h]
    np.fill_diagonal(D, 0.0)
    T_anch.append(D)
num = den = 0.0
L0a = np.sqrt(((P0a[Aa] - P0a[Ba]) ** 2).sum(1))
for T in T_anch:
    Tij = T[Aa, Ba]
    kp = np.isfinite(Tij) & (Tij > 30)
    num += (L0a[kp] * Tij[kp]).sum()
    den += (Tij[kp] ** 2).sum()
c_anch = num / den
span2a = ((P0a.max(0) - P0a.min(0)) ** 2).sum()
P0a_ss = (P0a * P0a).sum()

def solve_anchor_hour(T, X0):
    Tij = T[Aa, Ba]
    kp = np.isfinite(Tij) & (Tij > 30)
    A, B, D = Aa[kp], Ba[kp], c_anch * Tij[kp]
    wa = ANCHOR_W * len(A) / K / span2a

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
        Pc = P - P.mean(0)
        s = (Pc * P0a).sum() / P0a_ss
        dxy = Pc - s * P0a
        Ev += wa * (dxy * dxy).sum()
        grad += 2.0 * wa * dxy
        return Ev, grad.ravel()

    res = minimize(f, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "maxfun": 4000})
    return res.x.reshape(K, 2)

X_anch = []
Xa = P0a
for p in range(2):
    X_anch = []
    for h in range(24):
        Xa = solve_anchor_hour(T_anch[h], Xa)
        X_anch.append(Xa)
print("anchor-level layouts solved (pin targets)")
pin_idx = np.flatnonzero(anchor_sidx >= 0)
pin_nodes = anchor_sidx[pin_idx]

# ---- landmark MDS, identical machinery to animate_mds ----
rng = np.random.default_rng(7)
lm = rng.choice(N, size=min(N_LM, N), replace=False)
T_hours = []
for h in range(24):
    tt = elen / (sp[:, h] / 3.6)
    adj = coo_matrix((np.r_[tt, tt], (np.r_[ea, eb], np.r_[eb, ea])),
                     shape=(N, N)).tocsr()
    T_hours.append(dijkstra(adj, indices=lm))
print("dijkstra done for 24 hours")

A0 = np.repeat(lm, N)
B0 = np.tile(np.arange(N), len(lm))
# the anchor layout sets the scale; using its c keeps pins and stress
# pulling toward the same answer instead of fighting over size
c = c_anch
print(f"global scale {c:.2f} m/s ({c*3.6:.0f} kph equivalent, from anchors)")

span2 = ((Pg.max(0) - Pg.min(0)) ** 2).sum()
Pg_ss = (Pg * Pg).sum()
PIN_W = 5.0
EDGE_W = 0.15

def solve_hour(T, X0, Xt, h):
    Tij = T.ravel()
    kp = np.isfinite(Tij) & (Tij > 30) & (A0 != B0)
    A, B, D = A0[kp], B0[kp], c * Tij[kp]
    wa = ANCHOR_W * len(A) / N / span2
    wp = PIN_W * len(A) / len(pin_nodes) / span2
    # per-edge springs: each street edge wants its own travel-time length.
    # Landmarks + pins shape the city; springs tie neighbors together
    De = c * np.maximum(elen / (sp[:, h] / 3.6), 5.0)
    we = EDGE_W * len(A) / len(ea)

    def f(x):
        P = x.reshape(N, 2)
        grad = np.zeros_like(P)
        dp = P[A] - P[B]
        L = np.sqrt((dp * dp).sum(1) + 1e-9)
        r = (L - D) / D
        Ev = (r * r).sum()
        g = (2.0 * r / (D * L))[:, None] * dp
        grad[:, 0] += np.bincount(A, g[:, 0], minlength=N)
        grad[:, 0] -= np.bincount(B, g[:, 0], minlength=N)
        grad[:, 1] += np.bincount(A, g[:, 1], minlength=N)
        grad[:, 1] -= np.bincount(B, g[:, 1], minlength=N)
        dpe = P[ea] - P[eb]
        Le = np.sqrt((dpe * dpe).sum(1) + 1e-9)
        re = (Le - De) / De
        Ev += we * (re * re).sum()
        ge = (2.0 * we * re / (De * Le))[:, None] * dpe
        grad[:, 0] += np.bincount(ea, ge[:, 0], minlength=N)
        grad[:, 0] -= np.bincount(eb, ge[:, 0], minlength=N)
        grad[:, 1] += np.bincount(ea, ge[:, 1], minlength=N)
        grad[:, 1] -= np.bincount(eb, ge[:, 1], minlength=N)
        tmean = P.mean(0)
        Pc = P - tmean
        s = (Pc * Pg).sum() / Pg_ss
        dxy = Pc - s * Pg
        Ev += wa * (dxy * dxy).sum()
        grad += 2.0 * wa * dxy
        # pin solved anchor nodes to the anchor-level layout
        dpin = P[pin_nodes] - Xt
        Ev += wp * (dpin * dpin).sum()
        np.add.at(grad, pin_nodes, 2.0 * wp * dpin)
        return Ev, grad.ravel()

    res = minimize(f, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000, "maxfun": 6000})
    return res.x.reshape(N, 2), res.fun / len(A)

hours = []
X = Pg
t0 = time.time()
for p in range(2):
    out = []
    for h in range(24):
        X, stress = solve_hour(T_hours[h], X, X_anch[h][pin_idx], h)
        out.append(X)
        if p == 1:
            mean_r = np.sqrt((X ** 2).sum(1)).mean()
            r0 = np.sqrt((Pg ** 2).sum(1)).mean()
            print(f"  h={h:02d} stress/pair {stress:.4f} "
                  f"breath {mean_r/r0-1:+.1%} ({time.time()-t0:.0f}s)")
    hours = out

# ---- yardstick + export ----
yard10 = []
for h, X in enumerate(hours):
    Tij = T_hours[h].ravel()
    m = (Tij >= 540) & (Tij <= 660) & (A0 != B0)
    samp = rng.choice(np.flatnonzero(m), size=min(20000, m.sum()), replace=False)
    yard10.append(float(np.median(
        np.sqrt(((X[A0[samp]] - X[B0[samp]]) ** 2).sum(1)))))

edges_out = []
for i in range(len(ea)):
    edges_out.append({
        "u": int(ea[i]), "v": int(eb[i]), "ff": round(float(sp[i].max()), 1),
        "sp": [round(float(s), 1) for s in sp[i]],
        "obs": [True] * 24,
        "pts": np.round(geom[i], 1).ravel().tolist(),
    })
out = {
    "meta": {
        "mode": "day-mds-anchor-full", "scale_mps": float(c),
        "n_nodes": N, "n_anchors": K, "landmarks": int(len(lm)),
        "offsets_s": offsets.astype(int).tolist(), "yard10": yard10,
        "extent": [float(Pg[:, 0].min()), float(Pg[:, 0].max()),
                   float(Pg[:, 1].min()), float(Pg[:, 1].max())],
    },
    "nodes_flat": np.round(Pg, 1).ravel().tolist(),
    "node_hours": [np.round(Xh, 1).ravel().tolist() for Xh in hours],
    "edges": edges_out,
}
json.dump(out, open(OUT, "w"))
print(f"wrote {OUT} ({pathlib.Path(OUT).stat().st_size/1e6:.1f} MB)")
