"""2D time-space map: position intersections so Euclidean distance matches
network travel time (landmark-sampled stress MDS).

Unlike relax2d.py (each street's own drawn length ~ its travel time), this
encodes pairwise reachability: places linked by fast roads pull together even
if every street between them is slow. Classic travel-time cartogram.
"""

import argparse
import json
import pathlib
import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from relax2d import load_pruned

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DATA / "manhattan.graphml"))
    ap.add_argument("--out", default=str(DATA / "network_mds.json"))
    ap.add_argument("--landmarks", type=int, default=400)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--maxiter", type=int, default=3000)
    args = ap.parse_args()

    G = load_pruned(args.graph)
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    P0 = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])
    center = P0.mean(0)
    P0 -= center
    N = len(P0)
    print(f"{N} nodes, {len(G.edges)} edges")

    ea, eb, tt, speed, geom = [], [], [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        ea.append(idx[u]); eb.append(idx[v])
        sp = float(d.get("speed_kph", 40.0))
        speed.append(sp)
        tt.append(float(d["length"]) / (sp / 3.6))  # seconds
        if "geometry" in d:
            geom.append(np.array(d["geometry"].coords) - center)
        else:
            geom.append(np.array([P0[idx[u]], P0[idx[v]]]))
    ea = np.array(ea); eb = np.array(eb); tt = np.array(tt); speed = np.array(speed)

    adj = coo_matrix((np.concatenate([tt, tt]),
                      (np.concatenate([ea, eb]), np.concatenate([eb, ea]))),
                     shape=(N, N)).tocsr()
    rng = np.random.default_rng(7)
    lm = rng.choice(N, size=min(args.landmarks, N), replace=False)
    T = dijkstra(adj, indices=lm)            # (L, N) seconds
    print(f"{len(lm)} landmarks; time p50 {np.median(T[np.isfinite(T)]):.0f}s")

    A = np.repeat(lm, N)
    B = np.tile(np.arange(N), len(lm))
    Tij = T.ravel()
    keep = np.isfinite(Tij) & (Tij > 30) & (A != B)
    A, B, Tij = A[keep], B[keep], Tij[keep]

    # least-squares m-per-second scale from current geometry
    L0 = np.sqrt(((P0[A] - P0[B]) ** 2).sum(1))
    c = (L0 * Tij).sum() / (Tij * Tij).sum()
    D = c * Tij
    print(f"{len(A)} pairs; scale {c:.1f} m/s ({c*3.6:.0f} kph equivalent)")

    span2 = ((P0.max(0) - P0.min(0)) ** 2).sum()
    wa = args.anchor * len(A) / N / span2    # comparable to mean pair weight

    def f(x):
        P = x.reshape(N, 2)
        grad = np.zeros_like(P)
        dp = P[A] - P[B]
        L = np.sqrt((dp * dp).sum(1) + 1e-9)
        r = (L - D) / D
        E = (r * r).sum()
        g = (2.0 * r / (D * L))[:, None] * dp
        np.add.at(grad, A, g)
        np.add.at(grad, B, -g)
        dxy = P - P0
        E += wa * (dxy * dxy).sum()
        grad += 2.0 * wa * dxy
        return E, grad.ravel()

    E0, _ = f(P0.ravel())
    t0 = time.time()
    res = minimize(f, P0.ravel().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": args.maxiter, "maxfun": args.maxiter * 2})
    P = res.x.reshape(N, 2)
    print(f"{res.message.split(',')[0]} {time.time()-t0:.0f}s; "
          f"stress/pair {E0/len(A):.3f} -> {res.fun/len(A):.3f}")
    shift = np.sqrt(((P - P0) ** 2).sum(1))
    print(f"node shift mean {shift.mean():.0f}m max {shift.max():.0f}m")

    out_edges = []
    for i in range(len(ea)):
        pts0 = geom[i]
        z0 = pts0[:, 0] + 1j * pts0[:, 1]
        a1 = P[ea[i], 0] + 1j * P[ea[i], 1]
        b1 = P[eb[i], 0] + 1j * P[eb[i], 1]
        if abs(z0[-1] - z0[0]) > 1e-9:
            s = (b1 - a1) / (z0[-1] - z0[0])
            z1 = a1 + s * (z0 - z0[0])
        else:
            z1 = z0 + (a1 - z0[0])
        out_edges.append({
            "speed": round(float(speed[i]), 1),
            "hw": "", "warp": 0.0,
            "flat": np.round(pts0, 1).ravel().tolist(),
            "pos": np.round(np.column_stack([z1.real, z1.imag,
                                             np.zeros(len(z1))]), 1).ravel().tolist(),
        })

    out = {
        "meta": {"mode": "2d-mds", "scale_mps": float(c),
                 "n_points": N, "z_max": 0.0,
                 "extent": [float(P0[:, 0].min()), float(P0[:, 0].max()),
                            float(P0[:, 1].min()), float(P0[:, 1].max())]},
        "edges": out_edges,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
