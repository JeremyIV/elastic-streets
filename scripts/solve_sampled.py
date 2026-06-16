"""Importance-sampled all-pairs MDS -- a stochastic full spring model.

The ideal is a spring between EVERY pair of nodes (all-pairs MDS), rest length =
travel-time distance -- O(N^2) springs, too many. Instead we Horvitz-Thompson
sample: keep each pair with probability p = min(1, k/geo_dist) (drop long springs
more often) and weight survivors by 1/p. That gives (a) ~equal springs per
distance band -- every scale is constrained, so no landmark-undersampling spikes
-- and (b) an UNBIASED estimate of the full equal-weight all-pairs objective.

Sampling uses GEOGRAPHIC distance as a free proxy (HT is unbiased for any p>0);
true travel-time rest lengths come from a full Dijkstra per hour (~2s). Reads
per-edge speeds/geometry from an already-solved layout json (so it inherits that
run's imputation + free-flow calibration) and writes a renderable layout json.

Usage: solve_sampled.py in_layout.json out_layout.json
                        [--budget 1100000 --maxiter 1200 --seed 7]
"""
import argparse
import json
import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--budget", type=int, default=1_100_000, help="target spring count")
ap.add_argument("--maxiter", type=int, default=1200)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--wq", type=float, default=0.0,
                help="down-weight long springs by 1/dist^wq in the objective "
                     "(0 = equal-weight all-pairs, 2 = local like 1/D^2)")
ap.add_argument("--graphlocal", action="store_true",
                help="use the street graph for the local band (all edges) and "
                     "sample only pairs with geo>=lcut -- drops geo-close but "
                     "not-street-connected springs that distort fine structure")
ap.add_argument("--lcut", type=float, default=250.0,
                help="local-band cutoff (m): below it only street edges, above it sampled")
ap.add_argument("--ew", type=float, default=1.0,
                help="street-spring strength multiplier (1=consistent; >1 over-weights "
                     "street edges so they win locally -> smoother fine structure)")
args = ap.parse_args()

d = json.load(open(args.inp))
N = d["meta"]["n_nodes"]
P0 = np.array(d["nodes_flat"]).reshape(N, 2)
edges = d["edges"]
ea = np.array([e["u"] for e in edges]); eb = np.array([e["v"] for e in edges])
sp = np.array([e["sp"] for e in edges])                 # E x 24, kph


def plen(e):
    p = np.array(e["pts"]).reshape(-1, 2)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 1.0


length = np.maximum(np.array([plen(e) for e in edges]), 1.0)
print(f"{N} nodes, {len(edges)} edges", flush=True)

# ---- importance-sample pairs by geographic distance -------------------------
rng = np.random.default_rng(args.seed)
I, J = np.triu_indices(N, k=1)
geo = np.maximum(np.hypot(P0[I, 0] - P0[J, 0], P0[I, 1] - P0[J, 1]), 1e-9)
if args.graphlocal:
    # local band = the real street graph (street-connected, smooth); sample only
    # longer pairs for gross structure -> drops the geo-close-but-travel-far
    # swarm that distorts fine structure.
    far = geo >= args.lcut
    I, J, geo = I[far], J[far], geo[far]
    eg = np.maximum(np.hypot(P0[ea, 0] - P0[eb, 0], P0[ea, 1] - P0[eb, 1]), 1e-9)
    budget_far = max(args.budget - len(ea), 1)
else:
    eg = None
    budget_far = args.budget
lo, hi = 0.0, float(geo.max())                          # binary-search k for the budget
for _ in range(50):
    k = 0.5 * (lo + hi)
    if np.minimum(1.0, k / geo).sum() < budget_far:
        lo = k
    else:
        hi = k
k = 0.5 * (lo + hi)
p = np.minimum(1.0, k / geo)
keep = rng.random(len(geo)) < p
if args.graphlocal:                                     # street edges (p=1) + sampled far pairs
    A0 = np.concatenate([ea, I[keep]])
    B0 = np.concatenate([eb, J[keep]])
    WT = np.concatenate([args.ew * eg ** (-args.wq),
                         (1.0 / p[keep]) * geo[keep] ** (-args.wq)])
    print(f"springs: {len(ea)} street edges + {int(keep.sum())} sampled "
          f"(geo>={args.lcut:.0f} m, k={k:.0f} m) = {len(A0)}", flush=True)
else:
    A0, B0 = I[keep], J[keep]
    WT = (1.0 / p[keep]) * geo[keep] ** (-args.wq)
    print(f"sampled {len(A0)} springs (k={k:.0f} m)", flush=True)
WT = WT / WT.mean()                                     # mean-1 (numerics)
del I, J, geo, p, keep

# ---- per-hour all-pairs travel times for the sampled pairs -------------------
TT = np.zeros((24, len(A0)))
t0 = time.time()
for h in range(24):
    tt = length / (sp[:, h] / 3.6)
    adj = coo_matrix((np.concatenate([tt, tt]),
                      (np.concatenate([ea, eb]), np.concatenate([eb, ea]))),
                     shape=(N, N)).tocsr()
    Dm = dijkstra(adj)                                  # all-pairs N x N
    TT[h] = Dm[A0, B0]
    del Dm
print(f"all-pairs dijkstra x24: {time.time()-t0:.0f}s", flush=True)

# global scale c (pooled over hours): map travel-time onto the geographic scale
L0 = np.hypot(P0[A0, 0] - P0[B0, 0], P0[A0, 1] - P0[B0, 1])
num = den = 0.0
for h in range(24):
    ok = np.isfinite(TT[h]) & (TT[h] > 30)
    num += (WT[ok] * L0[ok] * TT[h][ok]).sum()
    den += (WT[ok] * TT[h][ok] ** 2).sum()
c = num / den
print(f"global scale {c:.2f} m/s ({c*3.6:.0f} kph)", flush=True)


def solve_hour(Th, X0):
    ok = np.isfinite(Th) & (Th > 30)
    A, B, D, W = A0[ok], B0[ok], c * Th[ok], WT[ok]

    def f(x):
        P = x.reshape(N, 2)
        grad = np.zeros_like(P)
        dp = P[A] - P[B]
        L = np.sqrt((dp * dp).sum(1) + 1e-9)
        diff = L - D                                    # equal-weight (unweighted) springs
        g = (W * 2.0 * diff / L)[:, None] * dp
        for col in (0, 1):
            grad[:, col] += np.bincount(A, g[:, col], minlength=N)
            grad[:, col] -= np.bincount(B, g[:, col], minlength=N)
        return float((W * diff * diff).sum()), grad.ravel()

    res = minimize(f, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": args.maxiter, "maxfun": args.maxiter * 2,
                            "ftol": 1e-7, "gtol": 1e-6})
    return res.x.reshape(N, 2)


def align(X):
    """Procrustes: rotate/reflect/translate X to the geographic map (gauge fix)."""
    Xc = X - X.mean(0)
    M = Xc.T @ (P0 - P0.mean(0))
    Uu, _, Vt = np.linalg.svd(M)
    return Xc @ (Uu @ Vt) + P0.mean(0)


X = P0.copy()
r0 = np.sqrt((P0 ** 2).sum(1)).mean()
t0 = time.time()
hours = []
for pas in range(2):
    out = []
    for h in range(24):
        X = solve_hour(TT[h], X)
        Xa = align(X)
        out.append(Xa)
        mr = np.sqrt(((Xa - Xa.mean(0)) ** 2).sum(1)).mean()
        print(f"  p{pas} h={h:02d} breath {mr/r0-1:+.1%} ({time.time()-t0:.0f}s)",
              flush=True)
    hours = out

meta = {"mode": "day-mds-sampled", "scale_mps": float(c),
        "city": d["meta"].get("city", "?"), "n_nodes": N,
        "yard10": float(c * 600.0), "n_springs": int(len(A0)),
        "extent": [float(P0[:, 0].min()), float(P0[:, 0].max()),
                   float(P0[:, 1].min()), float(P0[:, 1].max())]}
json.dump({"meta": meta,
           "nodes_flat": np.round(P0, 1).ravel().tolist(),
           "node_hours": [np.round(X, 1).ravel().tolist() for X in hours],
           "edges": edges}, open(args.out, "w"))
print(f"wrote {args.out}", flush=True)
