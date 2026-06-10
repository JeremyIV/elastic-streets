"""The city breathing: 24 hourly travel-time MDS layouts from observed speeds.

Joins the tracebase NYC Uber Movement matrix (segment x hour, mph, 0=missing)
to the graph by OSM way id, builds a weekday hour-of-day speed profile per
edge (free-flow fallback where unobserved), then solves the landmark MDS of
the travel-time MDS once per hour.

Two things make the animation honest and smooth:
- ONE global scale c (meters per second of travel time), fit across all
  hours pooled. Refitting per hour would normalize the breathing away;
  with c fixed, congestion makes travel times grow and the layout expand.
- Each hour warm-starts from the previous hour's layout (same landmarks,
  same anchor), so consecutive frames differ only where speeds differ.
"""

import argparse
import csv
import datetime
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


def load_pruned(path):
    """Undirected graph, dead-end ramp stubs pruned, largest component."""
    G = ox.load_graphml(path)
    G = nx.MultiGraph(ox.convert.to_undirected(G))
    ramp_classes = {"motorway", "motorway_link", "trunk", "trunk_link",
                    "primary_link", "secondary_link"}
    while True:
        drop = []
        for n in G.nodes:
            if G.degree(n) == 1:
                _, _, d = next(iter(G.edges(n, data=True)))
                hw = d.get("highway", "")
                if isinstance(hw, list):
                    hw = hw[0]
                if hw in ramp_classes:
                    drop.append(n)
        if not drop:
            break
        G.remove_nodes_from(drop)
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MPH_TO_KPH = 1.609344


def edge_hour_speeds(G, year=2019, month=1, data_dir=DATA):
    """Per-edge (graph order) x 24 hour-of-day mean weekday speed, kph.

    NaN where never observed.
    """
    mat = np.load(pathlib.Path(data_dir) / f"hourly_speed_mat_{year}_{month}.npz")["arr_0"]
    way_rows = defaultdict(list)
    with open(pathlib.Path(data_dir) / "road.csv") as fh:
        rdr = csv.reader(fh)
        next(rdr)
        for i, r in enumerate(rdr):
            way_rows[r[1]].append(i)

    ndays = mat.shape[1] // 24
    weekday = np.array([
        datetime.date(year, month, d + 1).weekday() < 5 for d in range(ndays)
    ])
    holidays = {datetime.date(2019, 1, 1), datetime.date(2019, 1, 21)}
    for d in range(ndays):
        if datetime.date(year, month, d + 1) in holidays:
            weekday[d] = False
    day_cols = np.flatnonzero(weekday)
    print(f"{len(day_cols)} weekdays used from {year}-{month:02d}")

    speeds = np.full((len(G.edges), 24), np.nan)
    covered = 0
    for ei, (u, v, k, d) in enumerate(G.edges(keys=True, data=True)):
        osmid = d.get("osmid")
        ids = osmid if isinstance(osmid, list) else [osmid]
        rows = [j for w in ids for j in way_rows.get(str(w), [])]
        if not rows:
            continue
        sub = mat[rows]                       # (r, ndays*24)
        sub = sub.reshape(len(rows), ndays, 24)[:, day_cols, :]
        obs = sub > 0
        cnt = obs.sum((0, 1))
        tot = np.where(obs, sub, 0.0).sum((0, 1))
        with np.errstate(invalid="ignore"):
            prof = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
        if np.isfinite(prof).any():
            speeds[ei] = prof * MPH_TO_KPH
            covered += 1
    print(f"observed profiles for {covered}/{len(G.edges)} edges")
    return speeds


def impute_speeds(sp, ff, midxy):
    """Fill unobserved edge-hours with spatially-imputed congestion.

    For each hour, an unobserved edge gets the inverse-distance-weighted mean
    congestion ratio (observed / free-flow) of its 8 nearest observed edges,
    applied to its own free-flow speed. Congestion is spatially correlated,
    so a gap in midtown crawls like midtown, not like the network average.
    """
    from scipy.spatial import cKDTree

    E = len(ff)
    ratio = sp / ff[:, None]
    filled = np.where(np.isfinite(sp), sp, np.nan)
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
        r_imp = (w * r_obs[j]).sum(1) / w.sum(1)
        r_imp = np.clip(r_imp, 0.1, 1.5)
        filled[miss, h] = ff[miss] * r_imp
    return np.clip(filled, 3.0, 110.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DATA / "manhattan.graphml"))
    ap.add_argument("--out", default=str(DATA / "day_mds.json"))
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--landmarks", type=int, default=400)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--planar", action="store_true",
                    help="forbid local foldovers (triangle orientation barrier)")
    ap.add_argument("--wtri", type=float, default=30.0)
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

    ea, eb, length, ff, geom = [], [], [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        ea.append(idx[u]); eb.append(idx[v])
        length.append(float(d["length"]))
        ff.append(float(d.get("speed_kph", 40.0)))
        if "geometry" in d:
            geom.append(np.array(d["geometry"].coords) - center)
        else:
            geom.append(np.array([P0[idx[u]], P0[idx[v]]]))
    ea = np.array(ea); eb = np.array(eb)
    length = np.array(length); ff = np.array(ff)

    sp = edge_hour_speeds(G, month=args.month, data_dir=args.data_dir)          # (E, 24) kph, NaN gaps
    sp = np.clip(sp, 3.0, 110.0)
    midxy = (P0[ea] + P0[eb]) / 2.0
    filled = impute_speeds(sp, ff, midxy)               # congestion-aware fill
    by_hour = np.nanmean(np.where(np.isfinite(sp), sp, np.nan), axis=0)
    print("network mean observed kph by hour:",
          np.round(by_hour, 1).tolist())

    # travel times per hour, and pairwise target distances from landmarks
    rng = np.random.default_rng(7)
    lm = rng.choice(N, size=min(args.landmarks, N), replace=False)

    T_hours = []
    for h in range(24):
        tt = length / (filled[:, h] / 3.6)               # seconds
        adj = coo_matrix((np.concatenate([tt, tt]),
                          (np.concatenate([ea, eb]), np.concatenate([eb, ea]))),
                         shape=(N, N)).tocsr()
        T_hours.append(dijkstra(adj, indices=lm))
    print("dijkstra done for 24 hours")

    A0 = np.repeat(lm, N)
    B0 = np.tile(np.arange(N), len(lm))

    # one global scale c over all hours pooled
    num = den = 0.0
    L0all = np.sqrt(((P0[A0] - P0[B0]) ** 2).sum(1))
    for T in T_hours:
        Tij = T.ravel()
        keep = np.isfinite(Tij) & (Tij > 30) & (A0 != B0)
        num += (L0all[keep] * Tij[keep]).sum()
        den += (Tij[keep] * Tij[keep]).sum()
    c = num / den
    print(f"global scale {c:.2f} m/s ({c*3.6:.0f} kph equivalent)")

    span2 = ((P0.max(0) - P0.min(0)) ** 2).sum()

    P0_ss = (P0 * P0).sum()

    # planar mode: triangulate the geographic fabric and forbid triangles
    # from collapsing below TAU of their area or flipping orientation, so
    # the deformation stays locally injective (no streets folding over)
    tri = a0 = None
    TAU = 0.2
    if args.planar:
        from scipy.spatial import Delaunay
        tri = Delaunay(P0).simplices
        u = P0[tri[:, 1]] - P0[tri[:, 0]]
        v = P0[tri[:, 2]] - P0[tri[:, 0]]
        a0 = 0.5 * (u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        neg = a0 < 0
        tri[neg] = tri[neg][:, [0, 2, 1]]
        a0 = np.abs(a0)
        L = lambda i, j: np.sqrt(((P0[tri[:, i]] - P0[tri[:, j]]) ** 2).sum(1))
        keep = (L(0, 1) < 600) & (L(1, 2) < 600) & (L(0, 2) < 600)
        tri, a0 = tri[keep], a0[keep]
        print(f"planar: {len(tri)} fabric triangles, barrier at {TAU} area")

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
            # similarity-invariant anchor: fit the best uniform scale s and
            # translation t of the geographic map to the current layout and
            # penalize only the residual shape difference. Exerts no force
            # against uniform expansion, so the layout's scale is set purely
            # by the travel-time targets (P0 is centered, so t = mean(P);
            # gradient via envelope theorem: s, t optimal -> direct term only)
            tmean = P.mean(0)
            Pc = P - tmean
            s = (Pc * P0).sum() / P0_ss
            dxy = Pc - s * P0
            E += wa * (dxy * dxy).sum()
            grad += 2.0 * wa * dxy

            if tri is not None and use_tri[0]:
                tu = P[tri[:, 1]] - P[tri[:, 0]]
                tv = P[tri[:, 2]] - P[tri[:, 0]]
                a = 0.5 * (tu[:, 0] * tv[:, 1] - tu[:, 1] * tv[:, 0])
                sviol = (TAU * a0 - a) / a0
                act = sviol > 0
                if act.any():
                    E += args.wtri * (sviol[act] ** 2).sum()
                    coef = -2.0 * args.wtri * sviol[act] / a0[act]
                    g1 = 0.5 * np.column_stack(
                        [tv[act][:, 1], -tv[act][:, 0]]) * coef[:, None]
                    g2 = 0.5 * np.column_stack(
                        [-tu[act][:, 1], tu[act][:, 0]]) * coef[:, None]
                    np.add.at(grad, tri[act, 1], g1)
                    np.add.at(grad, tri[act, 2], g2)
                    np.add.at(grad, tri[act, 0], -(g1 + g2))
            return E, grad.ravel()

        use_tri = [False]
        res = minimize(f, X0.ravel().copy(), jac=True, method="L-BFGS-B",
                       options={"maxiter": args.maxiter,
                                "maxfun": args.maxiter * 2})
        if tri is not None:
            # constrained polish: untangle flips from the converged layout
            use_tri[0] = True
            res = minimize(f, res.x, jac=True, method="L-BFGS-B",
                           options={"maxiter": 600, "maxfun": 1000})
        return res.x.reshape(N, 2), res.fun / len(A)

    hours = []
    X = P0
    t0 = time.time()
    # two passes: hour 0 of pass 1 starts from geography, which is far from
    # any congested layout; pass 2 re-solves with full warm context so the
    # 23 -> 0 wraparound is seamless too.
    for p in range(2):
        out = []
        for h in range(24):
            X, stress = solve_hour(T_hours[h], X)
            out.append(X)
            if p == 1:
                mean_r = np.sqrt((X ** 2).sum(1)).mean()
                r0 = np.sqrt((P0 ** 2).sum(1)).mean()
                fl = ""
                if tri is not None:
                    tu = X[tri[:, 1]] - X[tri[:, 0]]
                    tv = X[tri[:, 2]] - X[tri[:, 0]]
                    a = 0.5 * (tu[:, 0] * tv[:, 1] - tu[:, 1] * tv[:, 0])
                    fl = f" flips {(a <= 0).sum()}"
                print(f"  h={h:02d} stress/pair {stress:.3f} "
                      f"breath {mean_r/r0-1:+.1%}{fl} ({time.time()-t0:.0f}s)")
        hours = out

    # export: per-hour node positions + static edge geometry
    edges_out = []
    for i in range(len(ea)):
        edges_out.append({
            "u": int(ea[i]), "v": int(eb[i]),
            "ff": round(float(ff[i]), 1),
            "sp": [round(float(s), 1) for s in filled[i]],
            "obs": [bool(np.isfinite(x)) for x in sp[i]],
            "pts": np.round(geom[i], 1).ravel().tolist(),
        })
    out = {
        "meta": {
            "mode": "day-mds", "scale_mps": float(c), "month": args.month,
            "landmarks": int(len(lm)), "n_nodes": N,
            "extent": [float(P0[:, 0].min()), float(P0[:, 0].max()),
                       float(P0[:, 1].min()), float(P0[:, 1].max())],
        },
        "nodes_flat": np.round(P0, 1).ravel().tolist(),
        "node_hours": [np.round(X, 1).ravel().tolist() for X in hours],
        "edges": edges_out,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out} "
          f"({pathlib.Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
