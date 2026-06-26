"""Solve 24 hourly travel-time MDS layouts for a tile-collected city.

Speeds come from the live-traffic tiles, not Uber: per edge per hour,
  speed = FF[class] * r[class][category] * rel(edge)
where FF/r are the calibrated free-flow + category multipliers and rel is the
edge's free-flow relative to its class median (preserves within-class texture).
The MDS core (pooled global scale, similarity anchor, two-pass warm start) is
the same proven solver as animate_mds.py, run on projected coordinates.

Usage: solve_tiles.py city graph.graphml [--landmarks N] [--anchor A] [--maxiter M]
"""
import argparse
import json
import pathlib
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import osmnx as ox
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CATS = ["low", "moderate", "heavy", "severe"]
RAMP = {"motorway_link", "trunk_link", "primary_link", "secondary_link"}


def cls_of(d):
    h = d.get("highway", "?")
    return h[0] if isinstance(h, list) else h


def prune(Gu):
    """Drop dead-end ramp stubs iteratively, keep largest component."""
    while True:
        drop = []
        for n in Gu.nodes:
            if Gu.degree(n) == 1:
                _, _, d = next(iter(Gu.edges(n, data=True)))
                if cls_of(d) in RAMP:
                    drop.append(n)
        if not drop:
            break
        Gu.remove_nodes_from(drop)
    return Gu.subgraph(max(nx.connected_components(Gu), key=len)).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("graph")
    ap.add_argument("--landmarks", type=int, default=400)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--maxiter", type=int, default=3000)
    args = ap.parse_args()

    cal = json.load(open(DATA / f"{args.city}_calib.json"))
    FF, R, pooled = cal["FF_kmh"], cal["ratio"], cal["pooled_r"]
    cong = json.load(open(DATA / f"{args.city}_edge_cong.json"))
    pair_sev = {}
    for ei, (u, v, k) in enumerate(cong["edge_order"]):
        pr = (min(u, v), max(u, v))
        arr = pair_sev.get(pr)
        if arr is None:
            arr = pair_sev[pr] = np.full(24, -1, dtype=int)
        for h in range(24):
            s = cong["cong"].get(str(h), [None] * 0)
            if s and s[ei] > arr[h]:
                arr[h] = s[ei]

    G = ox.load_graphml(args.graph)
    Gp = ox.project_graph(G)
    Gu = prune(nx.MultiGraph(ox.convert.to_undirected(Gp)))
    nodes = list(Gu.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    P0 = np.array([[Gu.nodes[n]["x"], Gu.nodes[n]["y"]] for n in nodes])
    center = P0.mean(0)
    P0 -= center
    N = len(P0)
    print(f"{N} nodes, {Gu.number_of_edges()} edges (projected, pruned)")

    def ratio(cls, sev):
        rr = R.get(cls)
        cat = CATS[sev] if sev >= 0 else "low"
        if rr and rr.get(cat) is not None:
            return rr[cat]
        return pooled.get(cat, 0.6)

    def rlow(cls):
        rr = R.get(cls, {})
        return rr.get("low") or pooled.get("low", 0.8)

    # --- spatial low-congestion speed from the calibration probe links ---
    # Each edge's absolute free-flow (low-category) speed is kNN-interpolated
    # from nearby same-group probe samples, so the dense slow core is genuinely
    # slow and the fast highways genuinely fast. This single per-class FF was
    # the thing washing the core out to uniform mid-speed.
    import pyproj
    from scipy.spatial import cKDTree
    GRP = lambda c: 0 if c in {"motorway", "motorway_link", "trunk",
                               "trunk_link"} else 1
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", Gp.graph["crs"],
                                         always_xy=True)
    links = json.load(open(DATA / f"{args.city}_calib_links.json"))
    low = [l for l in links if l["cat"] == "low"]
    lX, lY = to_utm.transform([l["lon"] for l in low], [l["lat"] for l in low])
    lX = np.array(lX) - center[0]
    lY = np.array(lY) - center[1]
    lspd = np.array([l["kmh"] for l in low])
    lgrp = np.array([GRP(l["class"]) for l in low])
    trees = {}
    for g in (0, 1):
        m = lgrp == g
        if m.any():
            trees[g] = (cKDTree(np.column_stack([lX[m], lY[m]])), lspd[m])

    edges = list(Gu.edges(keys=True, data=True))
    mid = np.array([[(P0[idx[u]][0] + P0[idx[v]][0]) / 2,
                     (P0[idx[u]][1] + P0[idx[v]][1]) / 2]
                    for u, v, k, d in edges])
    egrp = np.array([GRP(cls_of(d)) for *_ , d in edges])
    lowsp = np.full(len(edges), np.nan)
    for g in (0, 1):
        m = egrp == g
        if not m.any() or g not in trees:
            continue
        tree, spd = trees[g]
        dd, jj = tree.query(mid[m], k=min(10, len(spd)))
        dd = np.atleast_2d(dd); jj = np.atleast_2d(jj)
        w = 1.0 / (dd + 30.0)
        lowsp[m] = (w * spd[jj]).sum(1) / w.sum(1)

    ea, eb, length, geom, sp, obs = [], [], [], [], [], []
    for ei, (u, v, k, d) in enumerate(edges):
        cls = cls_of(d)
        ls = lowsp[ei]
        if not np.isfinite(ls):
            ls = FF.get(cls, 40.0) * rlow(cls)
        rl = rlow(cls)
        sev24 = pair_sev.get((min(u, v), max(u, v)), np.full(24, -1, int))
        # absolute speed = (local low-congestion speed) x (category vs low)
        row = np.array([ls * ratio(cls, int(sev24[h])) / rl for h in range(24)])
        sp.append(np.clip(row, 3.0, 110.0))
        obs.append([bool(sev24[h] >= 0) for h in range(24)])
        ea.append(idx[u]); eb.append(idx[v])
        length.append(float(d["length"]))
        if "geometry" in d:
            geom.append(np.array(d["geometry"].coords) - center)
        else:
            geom.append(np.array([P0[idx[u]], P0[idx[v]]]))
    ea = np.array(ea); eb = np.array(eb); length = np.array(length)
    sp = np.array(sp)                          # (E, 24) kph
    ff_arr = np.array([FF.get(cls_of(d), 40.0)
                       for *_ , d in Gu.edges(keys=True, data=True)])

    print("network mean kph by hour:",
          np.round(sp.mean(0), 1).tolist())

    rng = np.random.default_rng(7)
    lm = rng.choice(N, size=min(args.landmarks, N), replace=False)
    T_hours = []
    for h in range(24):
        tt = length / (sp[:, h] / 3.6)
        adj = coo_matrix((np.concatenate([tt, tt]),
                          (np.concatenate([ea, eb]), np.concatenate([eb, ea]))),
                         shape=(N, N)).tocsr()
        T_hours.append(dijkstra(adj, indices=lm))
    print("dijkstra done", flush=True)

    A0 = np.repeat(lm, N)
    B0 = np.tile(np.arange(N), len(lm))
    L0all = np.sqrt(((P0[A0] - P0[B0]) ** 2).sum(1))
    num = den = 0.0
    for T in T_hours:
        Tij = T.ravel()
        keep = np.isfinite(Tij) & (Tij > 30) & (A0 != B0)
        num += (L0all[keep] * Tij[keep]).sum()
        den += (Tij[keep] * Tij[keep]).sum()
    c = num / den
    print(f"global scale {c:.2f} m/s ({c*3.6:.0f} kph)", flush=True)

    span2 = ((P0.max(0) - P0.min(0)) ** 2).sum()
    P0_ss = (P0 * P0).sum()

    def solve_hour(T, X0):
        Tij = T.ravel()
        keep = np.isfinite(Tij) & (Tij > 30) & (A0 != B0)
        A, B, D = A0[keep], B0[keep], c * Tij[keep]
        wa = args.anchor * len(A) / N / span2

        def f(x):
            P = x.reshape(N, 2)
            grad = np.zeros_like(P)
            dp = P[A] - P[B]
            L = np.sqrt((dp * dp).sum(1) + 1e-9)
            r = (L - D) / D
            E = (r * r).sum()
            g = (2.0 * r / (D * L))[:, None] * dp
            grad[:, 0] += np.bincount(A, g[:, 0], minlength=N)
            grad[:, 0] -= np.bincount(B, g[:, 0], minlength=N)
            grad[:, 1] += np.bincount(A, g[:, 1], minlength=N)
            grad[:, 1] -= np.bincount(B, g[:, 1], minlength=N)
            tmean = P.mean(0)
            Pc = P - tmean
            s = (Pc * P0).sum() / P0_ss
            dxy = Pc - s * P0
            E += wa * (dxy * dxy).sum()
            grad += 2.0 * wa * dxy
            return E, grad.ravel()

        res = minimize(f, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": args.maxiter,
                                "maxfun": args.maxiter * 2,
                                "ftol": 1e-6, "gtol": 1e-5})
        return res.x.reshape(N, 2), res.fun / len(A)

    hours = []
    X = P0.copy()
    t0 = time.time()
    r0 = np.sqrt((P0 ** 2).sum(1)).mean()
    for p in range(2):
        out = []
        for h in range(24):
            X, stress = solve_hour(T_hours[h], X)
            out.append(X.copy())
            mean_r = np.sqrt((X ** 2).sum(1)).mean()
            print(f"  p{p} h={h:02d} stress/pair {stress:.3f} "
                  f"breath {mean_r/r0-1:+.1%} ({time.time()-t0:.0f}s)",
                  flush=True)
        hours = out

    edges_out = []
    for i in range(len(ea)):
        edges_out.append({
            "u": int(ea[i]), "v": int(eb[i]),
            "ff": round(float(ff_arr[i]), 1),
            "sp": [round(float(s), 1) for s in sp[i]],
            "obs": obs[i],
            "pts": np.round(geom[i], 1).ravel().tolist(),
        })
    out = {
        "meta": {"mode": "day-mds-tiles", "scale_mps": float(c),
                 "city": args.city, "landmarks": int(len(lm)), "n_nodes": N,
                 "yard10": float(c * 600.0),   # 10-min ref = scale * 600 s
                 "extent": [float(P0[:, 0].min()), float(P0[:, 0].max()),
                            float(P0[:, 1].min()), float(P0[:, 1].max())]},
        "nodes_flat": np.round(P0, 1).ravel().tolist(),
        "node_hours": [np.round(X, 1).ravel().tolist() for X in hours],
        "edges": edges_out,
    }
    outpath = DATA / f"day_mds_{args.city}.json"
    json.dump(out, open(outpath, "w"))
    print(f"wrote {outpath} ({outpath.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
