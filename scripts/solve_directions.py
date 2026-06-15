"""Solve 24 hourly travel-time MDS layouts from CONTINUOUS Directions speeds.

Maps harvested per-segment (dist, dur) onto edge sample points (speed =
sum_dist/sum_dur per edge per hour), spatially imputes uncovered edges, then
runs the same MDS core as solve_tiles. Colors are best rendered with
COLOR_MODE=abs (continuous absolute speed -> dense slow core glows).

Usage: solve_directions.py city graph.graphml [--landmarks N --maxiter M]
"""
import argparse
import json
import pathlib
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import osmnx as ox
import pyproj
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAMP = {"motorway_link", "trunk_link", "primary_link", "secondary_link"}


def cls_of(d):
    h = d.get("highway", "?")
    return h[0] if isinstance(h, list) else h


def prune(Gu):
    while True:
        drop = [n for n in Gu.nodes if Gu.degree(n) == 1 and
                cls_of(next(iter(Gu.edges(n, data=True)))[2]) in RAMP]
        if not drop:
            break
        Gu.remove_nodes_from(drop)
    return Gu.subgraph(max(nx.connected_components(Gu), key=len)).copy()


def impute(sp, ff, midxy):
    """Fill unobserved edge-hours with IDW congestion ratio of 8 nearest covered."""
    ratio = sp / ff[:, None]
    filled = sp.copy()
    for h in range(24):
        obs = np.isfinite(ratio[:, h])
        miss = ~obs
        if not obs.any() or not miss.any():
            continue
        tree = cKDTree(midxy[obs])
        d, j = tree.query(midxy[miss], k=min(8, obs.sum()))
        d = np.atleast_2d(d); j = np.atleast_2d(j)
        w = 1.0 / (d + 50.0)
        r_obs = ratio[obs, h]
        r_imp = np.clip((w * r_obs[j]).sum(1) / w.sum(1), 0.1, 1.3)
        filled[miss, h] = ff[miss] * r_imp
    return np.clip(filled, 3.0, 110.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("city")
    ap.add_argument("graph")
    ap.add_argument("--landmarks", type=int, default=250)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--maxiter", type=int, default=1200)
    args = ap.parse_args()

    G = ox.load_graphml(args.graph)
    crs = G.graph.get("crs", "epsg:4326")
    sample_x = float(next(iter(G.nodes(data=True)))[1]["x"])
    Gp = G if abs(sample_x) > 1000 else ox.project_graph(G)
    crs = Gp.graph.get("crs", crs)
    Gu = prune(nx.MultiGraph(ox.convert.to_undirected(Gp)))
    nodes = list(Gu.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    P0 = np.array([[float(Gu.nodes[n]["x"]), float(Gu.nodes[n]["y"])]
                   for n in nodes])
    center = P0.mean(0)
    P0 -= center
    N = len(P0)
    print(f"{N} nodes, {Gu.number_of_edges()} edges", flush=True)

    edges = list(Gu.edges(keys=True, data=True))
    E = len(edges)
    geom = []
    for u, v, k, d in edges:
        if "geometry" in d:
            geom.append(np.array(d["geometry"].coords) - center)
        else:
            geom.append(np.array([P0[idx[u]], P0[idx[v]]]))

    # edge sample points (every ~25 m) for link matching
    sx, sy, se = [], [], []
    for ei, g in enumerate(geom):
        ls = LineString(g)
        npt = max(2, int(ls.length / 25) + 1)
        for s in np.linspace(0, ls.length, npt):
            p = ls.interpolate(s)
            sx.append(p.x); sy.append(p.y); se.append(ei)
    samp = np.column_stack([sx, sy])
    se = np.array(se)
    stree = cKDTree(samp)

    # accumulate dist/dur from harvested links onto edges, per hour
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    links = json.load(open(DATA / f"{args.city}_dir_links.json"))["links"]
    dacc = np.zeros((E, 24)); tacc = np.zeros((E, 24))
    for key, pts in links.items():
        if not pts:
            continue
        h = int(key.split("_")[1])
        a = np.array(pts)
        X, Y = to_xy.transform(a[:, 0], a[:, 1])
        q = np.column_stack([X - center[0], Y - center[1]])
        dist, j = stree.query(q, k=1, distance_upper_bound=35)
        ok = np.isfinite(dist)
        ei = se[j[ok]]
        np.add.at(dacc[:, h], ei, a[ok, 2])
        np.add.at(tacc[:, h], ei, a[ok, 3])
    sp = np.where(tacc > 0, dacc / np.maximum(tacc, 1e-6) * 3.6, np.nan)
    obs = np.isfinite(sp)
    cov = obs.any(1)
    print(f"coverage: {cov.sum()}/{E} edges ({100*cov.mean():.0f}%) covered "
          f"at >=1 hour; {100*obs.mean():.0f}% of edge-hours", flush=True)

    # free-flow per edge: observed fastest hour, else osmnx speed_kph
    ff = np.nanmax(np.where(obs, sp, np.nan), axis=1)
    spkph = np.array([float(d.get("speed_kph", 40.0)) for *_, d in edges])
    ff = np.where(np.isfinite(ff), ff, spkph)
    sp = impute(sp, ff, np.array([g.mean(0) for g in geom]))

    ea = np.array([idx[u] for u, v, k, d in edges])
    eb = np.array([idx[v] for u, v, k, d in edges])
    length = np.array([float(d["length"]) for *_, d in edges])
    print("network mean kph by hour:", np.round(sp.mean(0), 1).tolist(), flush=True)

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

    A0 = np.repeat(lm, N); B0 = np.tile(np.arange(N), len(lm))
    L0 = np.sqrt(((P0[A0] - P0[B0]) ** 2).sum(1))
    num = den = 0.0
    for T in T_hours:
        Tij = T.ravel()
        keep = np.isfinite(Tij) & (Tij > 30) & (A0 != B0)
        num += (L0[keep] * Tij[keep]).sum(); den += (Tij[keep] ** 2).sum()
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
            g = (2.0 * r / (D * L))[:, None] * dp
            for col in (0, 1):
                grad[:, col] += np.bincount(A, g[:, col], minlength=N)
                grad[:, col] -= np.bincount(B, g[:, col], minlength=N)
            tmean = P.mean(0); Pc = P - tmean
            s = (Pc * P0).sum() / P0_ss
            dxy = Pc - s * P0
            E_ = (r * r).sum() + wa * (dxy * dxy).sum()
            grad += 2.0 * wa * dxy
            return E_, grad.ravel()

        res = minimize(f, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": args.maxiter, "maxfun": args.maxiter * 2,
                                "ftol": 1e-6, "gtol": 1e-5})
        return res.x.reshape(N, 2), res.fun / len(A)

    X = P0.copy(); r0 = np.sqrt((P0 ** 2).sum(1)).mean(); t0 = time.time()
    hours = []
    for p in range(2):
        out = []
        for h in range(24):
            X, stress = solve_hour(T_hours[h], X)
            out.append(X.copy())
            mr = np.sqrt((X ** 2).sum(1)).mean()
            print(f"  p{p} h={h:02d} stress {stress:.3f} breath {mr/r0-1:+.1%} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        hours = out

    edges_out = [{"u": int(ea[i]), "v": int(eb[i]), "ff": round(float(ff[i]), 1),
                  "sp": [round(float(s), 1) for s in sp[i]],
                  "obs": [bool(o) for o in obs[i]],
                  "pts": np.round(geom[i], 1).ravel().tolist()} for i in range(E)]
    out = {"meta": {"mode": "day-mds-directions", "scale_mps": float(c),
                    "city": args.city, "landmarks": int(len(lm)), "n_nodes": N,
                    "yard10": float(c * 600.0),
                    "extent": [float(P0[:, 0].min()), float(P0[:, 0].max()),
                               float(P0[:, 1].min()), float(P0[:, 1].max())]},
           "nodes_flat": np.round(P0, 1).ravel().tolist(),
           "node_hours": [np.round(X, 1).ravel().tolist() for X in hours],
           "edges": edges_out}
    outpath = DATA / f"day_mds_{args.city}_dir.json"
    json.dump(out, open(outpath, "w"))
    print(f"wrote {outpath} ({outpath.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
