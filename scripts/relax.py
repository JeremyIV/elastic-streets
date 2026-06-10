"""Warp the street network by free-flow speed and embed it in 3D.

The metric: each street's rest length is its geographic length times
warp(v) = (v_ref / v) ** alpha. Streets at the reference speed keep their
length, faster streets contract toward nothing, slower ones grow.

Two-phase embedding, so the result is coherent topography rather than
per-street buckling noise:

Phase A (in-plane): relax xy with contraction only (rest = min(warped, geo)).
Fast corridors pull the footprint inward; an elastic "city fabric" (kNN
coupling of nearby points' displacements) keeps the deformation smooth and a
weak anchor keeps the city recognizable. Nothing can buckle here.

Phase B (height): freeze xy and lift. Each segment has a grade it needs,
s = sqrt(rest^2 - Lxy^2) / Lxy — the slope that makes its 3D arc length match
its warped rest length (0 where the plane already holds it). The height field
is the slope-limited distance transform: h(x) = the height you reach climbing
from the wants-to-stay-flat set with every street as a ramp at its grade
(multi-source Dijkstra, edge cost s * length). Neighborhoods far from fast
roads rise into hills; fast corridors stay pinned as valleys. The field is
projected onto a ~120 m control grid and lightly smoothed, which removes
cross-basin cliffs and block-scale noise.

(A pure energy-minimizing 3D embedding was tried first and is instructive:
this metric concentrates excess length at block scale, so a true elastic
relaxation either crumples every street individually — fuzz, not topography —
or, with any meaningful smoothness term, finds flat-with-strain cheaper than
coherent hills. The distance transform is the legible upper envelope of the
same quantity: how much length the plane fails to hold, and where.)
"""

import argparse
import json
import pathlib
import time

import networkx as nx
import numpy as np
import osmnx as ox
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def resample_polyline(coords: np.ndarray, target: float):
    seg = np.diff(coords, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    total = seglen.sum()
    if total <= 0:
        return coords[[0, -1]], np.array([0.0])
    n = max(1, int(round(total / target)))
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    s = np.linspace(0.0, total, n + 1)
    x = np.interp(s, cum, coords[:, 0])
    y = np.interp(s, cum, coords[:, 1])
    return np.column_stack([x, y]), np.full(n, total / n)


def build_system(G, seg_target, v_ref, alpha):
    node_ids = list(G.nodes)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    pts = [np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_ids])]
    n_pts = len(node_ids)

    spring_a, spring_b, spring_rest, spring_geo = [], [], [], []
    chains = []

    for u, v, k, d in G.edges(keys=True, data=True):
        if "geometry" in d:
            coords = np.array(d["geometry"].coords)
        else:
            coords = np.array(
                [[G.nodes[u]["x"], G.nodes[u]["y"]], [G.nodes[v]["x"], G.nodes[v]["y"]]]
            )
        rp, lens = resample_polyline(coords, seg_target)
        speed = float(d.get("speed_kph", 40.0))
        warp = (v_ref / max(speed, 1.0)) ** alpha

        k_interior = len(rp) - 2
        interior_idx = np.arange(n_pts, n_pts + k_interior)
        if k_interior > 0:
            pts.append(rp[1:-1])
            n_pts += k_interior
        chain = np.concatenate([[node_index[u]], interior_idx, [node_index[v]]])

        hw = d.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        chains.append((chain, speed, str(hw), warp))

        spring_a.append(chain[:-1])
        spring_b.append(chain[1:])
        spring_geo.append(lens)
        spring_rest.append(lens * warp)

    return {
        "P0": np.vstack(pts),
        "a": np.concatenate(spring_a),
        "b": np.concatenate(spring_b),
        "rest": np.concatenate(spring_rest),
        "geo": np.concatenate(spring_geo),
        "chains": chains,
    }


def fabric_pairs(P0, radius):
    tree = cKDTree(P0)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    fi, fj = pairs[:, 0], pairs[:, 1]
    pr = np.sqrt(((P0[fi] - P0[fj]) ** 2).sum(1))
    wlin = 1.0 - pr / radius
    return fi, fj, wlin


def solve_plane(sys, fi, fj, wlin, w_fabric, w_anchor, maxiter):
    """Phase A: 2D relaxation, contraction only."""
    P0 = sys["P0"]
    N = len(P0)
    a, b = sys["a"], sys["b"]
    rest = np.minimum(sys["rest"], sys["geo"])  # expansion deferred to phase B
    w = 1.0 / np.maximum(rest, 1.0)
    wf = w_fabric * wlin
    rel0 = P0[fi] - P0[fj]
    eps = 1e-9

    def f(x):
        P = x.reshape(N, 2)
        grad = np.zeros_like(P)

        d = P[a] - P[b]
        L = np.sqrt((d * d).sum(1) + eps)
        dl = L - rest
        E = (w * dl * dl).sum()
        g = (2.0 * w * dl / L)[:, None] * d
        np.add.at(grad, a, g)
        np.add.at(grad, b, -g)

        du = P[fi] - P[fj] - rel0
        E += (wf * (du * du).sum(1)).sum()
        gu = 2.0 * wf[:, None] * du
        for c in range(2):
            grad[:, c] += np.bincount(fi, gu[:, c], minlength=N)
            grad[:, c] -= np.bincount(fj, gu[:, c], minlength=N)

        dxy = P - P0
        E += w_anchor * (dxy * dxy).sum()
        grad += 2.0 * w_anchor * dxy
        return E, grad.ravel()

    t0 = time.time()
    res = minimize(f, P0.ravel().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": maxiter * 2})
    P = res.x.reshape(N, 2)
    shift = np.sqrt(((P - P0) ** 2).sum(1))
    print(f"phase A: {res.message.split(',')[0]} {time.time()-t0:.0f}s; "
          f"mean shift {shift.mean():.0f}m, max {shift.max():.0f}m")
    return P


def solve_height(sys, Pxy, smooth_iters, cap, grid_step=120.0):
    """Phase B: the height field, on a coarse control grid (Gaussian-
    interpolated to street points so block-scale noise is impossible)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    N = len(Pxy)
    a, b = sys["a"], sys["b"]
    rest = sys["rest"]
    dxy = Pxy[a] - Pxy[b]
    Lxy2 = (dxy * dxy).sum(1)
    Lxy = np.sqrt(Lxy2)

    # --- control grid over the contracted footprint ---
    lo = Pxy.min(0) - grid_step
    hi = Pxy.max(0) + grid_step
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], grid_step),
                         np.arange(lo[1], hi[1], grid_step))
    gridxy = np.column_stack([gx.ravel(), gy.ravel()])
    pt_tree = cKDTree(Pxy)
    near = pt_tree.query_ball_point(gridxy, r=grid_step * 1.5)
    gridxy = gridxy[[len(n) > 0 for n in near]]
    K = len(gridxy)

    # interpolation B: street point heights = B @ grid heights
    gtree = cKDTree(gridxy)
    sigma = 0.75 * grid_step
    nbrs = gtree.query_ball_point(Pxy, r=grid_step * 1.9)
    ri, ci, vi = [], [], []
    for i, js in enumerate(nbrs):
        js = np.asarray(js, dtype=int)
        if len(js) == 0:
            d, j = gtree.query(Pxy[i])
            js = np.array([j])
        d2 = ((gridxy[js] - Pxy[i]) ** 2).sum(1)
        wgt = np.exp(-d2 / (2 * sigma * sigma))
        wgt /= wgt.sum()
        ri.append(np.full(len(js), i)); ci.append(js); vi.append(wgt)
    B = coo_matrix((np.concatenate(vi), (np.concatenate(ri), np.concatenate(ci))),
                   shape=(N, K)).tocsr()
    Bt = B.T.tocsr()

    gpairs = gtree.query_pairs(r=grid_step * 1.6, output_type="ndarray")
    gi, gj = gpairs[:, 0], gpairs[:, 1]

    # The height field: a slope-limited distance transform. Each segment can
    # climb at most s = sqrt(rest^2 - Lxy^2)/Lxy (the grade that makes its 3D
    # length match its warped rest length; 0 where the plane already holds
    # it). h(x) = max height reachable from the wants-to-stay-flat set
    # climbing every street at its grade — neighborhoods far from fast roads
    # rise, fast corridors stay pinned to the ground. Computed by multi-source
    # Dijkstra with edge cost s * length, then projected onto the coarse grid
    # (kills cross-basin cliffs) and lightly smoothed.
    slope = np.sqrt(np.maximum(rest * rest - Lxy2, 0.0)) / np.maximum(Lxy, 1.0)
    cost = slope * Lxy
    adj = coo_matrix((np.concatenate([cost, cost]),
                      (np.concatenate([a, b]), np.concatenate([b, a]))),
                     shape=(N, N)).tocsr()
    pt_slope = np.bincount(a, slope, minlength=N) + np.bincount(b, slope, minlength=N)
    pt_deg = np.bincount(a, minlength=N) + np.bincount(b, minlength=N)
    pt_slope /= np.maximum(pt_deg, 1)
    sources = np.flatnonzero(pt_slope < 0.05)
    if len(sources) == 0:
        sources = np.array([int(np.argmin(pt_slope))])
    h0 = dijkstra(adj, indices=sources, min_only=True)
    h0[~np.isfinite(h0)] = 0.0
    if cap > 0:
        h0 = cap * np.tanh(h0 / cap)   # soft-cap lonely towers
    print(f"grid {K} nodes; {len(sources)} flat sources, "
          f"h0 max {h0.max():.0f}m mean {h0.mean():.0f}m")

    c = (Bt @ h0) / np.maximum(Bt @ np.ones(N), 1e-9)
    # Jacobi smoothing on the grid neighbor graph
    nb_i = np.concatenate([gi, gj])
    nb_j = np.concatenate([gj, gi])
    deg = np.bincount(nb_i, minlength=K).astype(float)
    deg = np.maximum(deg, 1)
    for _ in range(smooth_iters):
        mean_nbr = np.bincount(nb_i, c[nb_j], minlength=K) / deg
        c = 0.5 * c + 0.5 * mean_nbr
    h = B @ c
    print(f"phase B: h mean {h.mean():.0f}m, p95 {np.percentile(h,95):.0f}m, "
          f"max {h.max():.0f}m")
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DATA / "manhattan.graphml"))
    ap.add_argument("--out", default=str(DATA / "network.json"))
    ap.add_argument("--seg", type=float, default=35.0, help="segment target (m)")
    ap.add_argument("--alpha", type=float, default=1.6, help="warp exponent")
    ap.add_argument("--vref", type=float, default=40.0,
                    help="reference speed kph (0 = 30th percentile)")
    ap.add_argument("--anchor", type=float, default=5e-5, help="xy anchor weight")
    ap.add_argument("--fabric", type=float, default=0.01, help="phase A fabric weight")
    ap.add_argument("--fabric-r", type=float, default=180.0, help="fabric radius (m)")
    ap.add_argument("--smooth", type=int, default=2, help="phase B smoothing iters")
    ap.add_argument("--cap", type=float, default=600.0, help="soft height cap (m)")
    ap.add_argument("--maxiter", type=int, default=3000)
    args = ap.parse_args()

    print("Loading graph...")
    G = ox.load_graphml(args.graph)
    G = nx.MultiGraph(ox.convert.to_undirected(G))

    # prune dangling highway-ramp tails (bridges clipped at the borough
    # boundary leave floating stubs)
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
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"{len(G.nodes)} nodes, {len(G.edges)} edges (undirected, pruned)")

    speeds = np.array([d.get("speed_kph", 40.0) for _, _, d in G.edges(data=True)],
                      dtype=float)
    v_ref = args.vref if args.vref > 0 else float(np.percentile(speeds, 30))
    print(f"speeds p30/median/max: {np.percentile(speeds,30):.0f}/"
          f"{np.median(speeds):.0f}/{speeds.max():.0f} kph; "
          f"v_ref={v_ref:.1f}, alpha={args.alpha}")

    sys = build_system(G, args.seg, v_ref, args.alpha)
    P0 = sys["P0"]
    P0 -= P0.mean(0)
    N = len(P0)
    print(f"{N} points, {len(sys['rest'])} springs")

    fi, fj, wlin = fabric_pairs(P0, args.fabric_r)
    print(f"{len(fi)} fabric pairs (r={args.fabric_r:.0f}m)")

    Pxy = solve_plane(sys, fi, fj, wlin, args.fabric, args.anchor, args.maxiter)
    h = solve_height(sys, Pxy, args.smooth, args.cap)

    P = np.column_stack([Pxy, h])

    # fidelity: how much of the warped metric the embedding realizes
    sa, sb = sys["a"], sys["b"]
    L3 = np.sqrt(((P[sa] - P[sb]) ** 2).sum(1))
    for name, m in (("contracting", sys["rest"] < 0.97 * sys["geo"]),
                    ("neutral", abs(sys["rest"] - sys["geo"]) <= 0.03 * sys["geo"]),
                    ("growing", sys["rest"] > 1.03 * sys["geo"])):
        if m.any():
            print(f"fidelity {name:11s}: realized/target length "
                  f"{L3[m].sum()/sys['rest'][m].sum():.2f} "
                  f"({m.sum()} segments)")

    edges_out = []
    for chain, speed, hw, warp in sys["chains"]:
        edges_out.append({
            "speed": round(speed, 1),
            "hw": hw,
            "warp": round(warp, 3),
            "flat": np.round(P0[chain], 1).ravel().tolist(),
            "pos": np.round(P[chain], 1).ravel().tolist(),
        })

    out = {
        "meta": {
            "alpha": args.alpha, "v_ref": v_ref, "anchor": args.anchor,
            "fabric": args.fabric, "smooth": args.smooth, "cap": args.cap,
            "n_points": N, "z_max": float(h.max()),
            "extent": [float(P0[:, 0].min()), float(P0[:, 0].max()),
                       float(P0[:, 1].min()), float(P0[:, 1].max())],
        },
        "edges": edges_out,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out} ({pathlib.Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
