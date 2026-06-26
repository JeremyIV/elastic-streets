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
from scipy.sparse import coo_matrix, triu
from scipy.sparse.csgraph import dijkstra

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--budget", type=int, default=1_100_000, help="target spring count")
ap.add_argument("--maxiter", type=int, default=1200)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--wq", type=float, default=1.0,
                help="down-weight long springs by 1/dist^wq in the objective "
                     "(0 = equal-weight all-pairs, 2 = local like 1/D^2; default 1)")
ap.add_argument("--graphlocal", action=argparse.BooleanOptionalAction, default=True,
                help="use the street graph for the local band (all edges) and "
                     "sample only pairs with geo>=lcut -- drops geo-close but "
                     "not-street-connected springs that distort fine structure "
                     "(default on; --no-graphlocal for pure all-pairs sampling)")
ap.add_argument("--lcut", type=float, default=250.0,
                help="local-band cutoff (m): below it only street edges, above it sampled")
ap.add_argument("--ew", type=float, default=100.0,
                help="street-spring strength multiplier (1=consistent; >1 over-weights "
                     "street edges so they win locally -> smoother fine; default 100)")
ap.add_argument("--hops", type=int, default=2,
                help="local stratum = all node pairs within this many street-hops "
                     "(1=immediate edges, 2=neighbors-of-neighbors; default 2)")
ap.add_argument("--anchor", type=float, default=0.0,
                help="similarity-invariant Procrustes pull toward geographic shape "
                     "(0=off; higher = more 'looks like normal Manhattan')")
ap.add_argument("--corr", action=argparse.BooleanOptionalAction, default=True,
                help="correlated far-pair sampling: each node gets one shared "
                     "latent value, nearby nodes reuse it so they pick the SAME "
                     "far 'hub' partners -> aligned far-field, less street-to-street "
                     "jitter. Same per-distance marginals (HT weights unchanged). "
                     "--no-corr = independent per-pair Bernoulli (old behavior)")
ap.add_argument("--landmarks", type=int, default=0,
                help="if >0, the far field is K shared landmarks (each sprung to "
                     "EVERY node) instead of sampled pairs -- deterministic, "
                     "spatially coherent far field (the landmark solver) combined "
                     "with the local street mesh. Smooth far field + constrained "
                     "short edges; overrides --corr/sampling. Try --wq 2 (1/D^2) "
                     "to match the old _dir look.")
ap.add_argument("--fw", type=float, default=1.0,
                help="far-spring weight multiplier (long-distance springs only). "
                     "<1 weakens the far field so the local street mesh dominates "
                     "(smoother, less jaggy); local springs AND the anchor are held "
                     "at their fw=1 strength, so this is a clean local-vs-far dial.")
ap.add_argument("--shape", type=float, default=0.0,
                help="as-similar-as-possible angle term: penalize each junction's "
                     "street fan deviating from a rotation+scale of its geographic "
                     "shape. Preserves local angles (kills street-to-street jitter) "
                     "while leaving local scale free (breathing intact). 0=off; "
                     "sweep up until breathing starts to flatten.")
ap.add_argument("--init", default=None,
                help="warm-start: seed each hour from this layout json's node_hours. "
                     "Near-optimal init -> L-BFGS converges in far fewer iters for the "
                     "SAME minimum. Ideal for sweeping a small param (e.g. --shape) off "
                     "an existing solve; pair with --passes 1.")
ap.add_argument("--passes", type=int, default=2,
                help="solve passes over the 24 hours (2 from cold start; 1 is enough "
                     "when warm-started via --init)")
args = ap.parse_args()

d = json.load(open(args.inp))
N = d["meta"]["n_nodes"]
P0 = np.array(d["nodes_flat"]).reshape(N, 2)
span2 = ((P0.max(0) - P0.min(0)) ** 2).sum()
P0_ss = (P0 * P0).sum()
edges = d["edges"]
ea = np.array([e["u"] for e in edges]); eb = np.array([e["v"] for e in edges])
sp = np.array([e["sp"] for e in edges])                 # E x 24, kph


def plen(e):
    p = np.array(e["pts"]).reshape(-1, 2)
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 1.0


length = np.maximum(np.array([plen(e) for e in edges]), 1.0)
print(f"{N} nodes, {len(edges)} edges", flush=True)

# --- shape/angle (ASAP) precompute: real street edges, both directions. Skip
# near-zero-chord loop edges (their direction is meaningless). E0 = geographic
# edge vectors; SDEG = per-node incident count; used by the --shape term. ---
_Z0 = P0[:, 0] + 1j * P0[:, 1]
_e0 = _Z0[eb] - _Z0[ea]
_sk = np.abs(_e0) > 1.0
SU = np.concatenate([ea[_sk], eb[_sk]])
SV = np.concatenate([eb[_sk], ea[_sk]])
E0 = np.concatenate([_e0[_sk], -_e0[_sk]])
E0N2 = np.maximum(E0.real * E0.real + E0.imag * E0.imag, 1e-12)
SDEG = np.maximum(np.bincount(SU, minlength=N), 1)

# ---- importance-sample pairs by geographic distance -------------------------
rng = np.random.default_rng(args.seed)
I, J = np.triu_indices(N, k=1)
geo = np.maximum(np.hypot(P0[I, 0] - P0[J, 0], P0[I, 1] - P0[J, 1]), 1e-9)
if args.graphlocal:
    # local stratum = node pairs within args.hops street-hops (street-connected,
    # smooth); sample only longer pairs for gross structure -> drops the
    # geo-close-but-travel-far swarm that distorts fine structure.
    if args.hops <= 1:
        la, lb = ea, eb
    else:
        A = coo_matrix((np.ones(2 * len(ea)),
                        (np.concatenate([ea, eb]), np.concatenate([eb, ea]))),
                       shape=(N, N)).tocsr()
        reach = A.copy(); acc = A.copy()
        for _ in range(args.hops - 1):
            acc = acc @ A
            reach = reach + acc
        R = triu(reach, k=1).tocoo()
        la, lb = R.row, R.col
    lg = np.maximum(np.hypot(P0[la, 0] - P0[lb, 0], P0[la, 1] - P0[lb, 1]), 1e-9)
    far = geo >= args.lcut
    I, J, geo = I[far], J[far], geo[far]
    budget_far = max(args.budget - len(la), 1)
else:
    budget_far = args.budget
if args.landmarks > 0:
    # FAR FIELD = shared landmarks. K random nodes, each sprung to EVERY node.
    # Every node references the SAME landmarks, so the far-field pull is spatially
    # coherent -- no per-node sampling noise (independent -> jitter) and no shared-
    # latent hubs (corr -> low-freq spikes). This is the classic landmark solver;
    # the local street mesh below supplies the short-edge constraints the pure
    # landmark solver lacked (those gaps were its spike source). --wq 2 -> 1/D^2.
    Kl = min(args.landmarks, N)
    lm = rng.choice(N, size=Kl, replace=False)
    Af = np.repeat(lm, N)
    Bf = np.tile(np.arange(N), Kl)
    m = Af != Bf
    Af, Bf = Af[m], Bf[m]
    gf = np.maximum(np.hypot(P0[Af, 0] - P0[Bf, 0], P0[Af, 1] - P0[Bf, 1]), 1e-9)
    WTf = gf ** (-args.wq)
    print(f"landmarks: {Kl} x {N} = {len(Af)} far springs", flush=True)
else:
    lo, hi = 0.0, float(geo.max())                      # binary-search k for the budget
    for _ in range(50):
        k = 0.5 * (lo + hi)
        if np.minimum(1.0, k / geo).sum() < budget_far:
            lo = k
        else:
            hi = k
    k = 0.5 * (lo + hi)
    p = np.minimum(1.0, k / geo)
    if args.corr:
        # Correlated sampling (variance reduction): give every node one shared latent
        # value v ~ U(0,1) and keep a pair iff v_i + v_j < t(p), the p-quantile of the
        # sum of two uniforms (triangular on [0,2]; closed form below). Per-pair this is
        # still P[keep] = p exactly, so the HT weights 1/p and the per-distance density
        # are unchanged -- but because each node reuses its v across all its springs,
        # low-v nodes act as shared "hubs" everyone springs to, so neighboring nodes
        # pick the same far partners. Their far-field pulls align -> the random
        # street-to-street jitter of independent sampling goes away.
        vnode = rng.random(N)
        t = np.where(p <= 0.5, np.sqrt(2.0 * p),
                     2.0 - np.sqrt(2.0 * np.maximum(1.0 - p, 0.0)))
        keep = (vnode[I] + vnode[J]) < t
    else:
        keep = rng.random(len(geo)) < p
    Af, Bf = I[keep], J[keep]
    WTf = (1.0 / p[keep]) * geo[keep] ** (-args.wq)
    print(f"sampled {int(keep.sum())} far springs (k={k:.0f} m)", flush=True)
if args.graphlocal:                                     # local mesh + far springs
    A0 = np.concatenate([la, Af])
    B0 = np.concatenate([lb, Bf])
    WT = np.concatenate([args.ew * lg ** (-args.wq), WTf])
    print(f"  + {len(la)} local (<={args.hops} hops) = {len(A0)} springs", flush=True)
else:
    A0, B0, WT = Af, Bf, WTf
WT = WT / WT.mean()                                     # mean-1 (numerics)
# split local vs far so --fw re-weights ONLY the far springs. WTG keeps the gold
# (fw=1) weights; the anchor is normalized from WTG, so weakening the far field
# moves neither the local mesh nor the anchor -- a clean local-vs-far dial.
n_local = len(la) if args.graphlocal else 0
is_far = np.zeros(len(WT), dtype=bool)
is_far[n_local:] = True
WTG = WT.copy()
if args.fw != 1.0:
    WT = WT.copy()
    WT[is_far] *= args.fw
    print(f"far-spring weight x{args.fw} (local mesh + anchor held at fw=1)", flush=True)
del I, J, geo

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
    ok = np.isfinite(TT[h]) & (TT[h] > 1)
    num += (WT[ok] * L0[ok] * TT[h][ok]).sum()
    den += (WT[ok] * TT[h][ok] ** 2).sum()
c = num / den
print(f"global scale {c:.2f} m/s ({c*3.6:.0f} kph)", flush=True)


def solve_hour(Th, X0):
    ok = np.isfinite(Th) & (Th > 1)
    A, B, D, W = A0[ok], B0[ok], c * Th[ok], WT[ok]
    Wg = WTG[ok]                                          # gold weights for the anchor

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
        E_ = float((W * diff * diff).sum())
        if args.anchor > 0.0 or args.shape > 0.0:
            Wd2 = float((Wg * D * D).sum())          # distance-energy scale (shared)
        if args.anchor > 0:                          # Procrustes pull to geographic
            tmean = P.mean(0); Pc = P - tmean
            sa = (Pc * P0).sum() / P0_ss
            dxy = Pc - sa * P0
            wa = args.anchor * Wd2 / (N * span2)
            E_ += wa * float((dxy * dxy).sum())
            grad += 2.0 * wa * dxy
        if args.shape > 0:                           # ASAP: each junction's street fan
            # should be a rotation+scale of its geographic fan (angles preserved,
            # local scale free). svec = per-node envelope-optimal similarity (mean of
            # per-edge complex ratios e/e0), so we differentiate holding it fixed.
            Zc = P[:, 0] + 1j * P[:, 1]
            e = Zc[SV] - Zc[SU]
            ratio = e / E0
            svec = (np.bincount(SU, ratio.real, minlength=N)
                    + 1j * np.bincount(SU, ratio.imag, minlength=N)) / SDEG
            r = e - svec[SU] * E0                     # residual after best local similarity
            we = (args.shape * Wd2 / max(len(SU), 1)) / E0N2
            E_ += float((we * (r.real * r.real + r.imag * r.imag)).sum())
            gr = 2.0 * we * r
            grad[:, 0] += (np.bincount(SV, gr.real, minlength=N)
                           - np.bincount(SU, gr.real, minlength=N))
            grad[:, 1] += (np.bincount(SV, gr.imag, minlength=N)
                           - np.bincount(SU, gr.imag, minlength=N))
        return E_, grad.ravel()

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


INIT = None
if args.init:
    INIT = [np.array(x).reshape(N, 2) for x in json.load(open(args.init))["node_hours"]]
    print(f"warm-start from {args.init}", flush=True)
X = P0.copy()
r0 = np.sqrt((P0 ** 2).sum(1)).mean()
t0 = time.time()
hours = []
for pas in range(args.passes):
    out = []
    for h in range(24):
        X0 = INIT[h] if (INIT is not None and pas == 0) else X
        X = solve_hour(TT[h], X0)
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
