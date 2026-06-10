"""2D elastic street map: move intersections so drawn street lengths match
travel times.

Control points are the graph nodes (intersections) only — interior street
geometry is carried along by a per-edge similarity transform — so an edge's
drawn length is the chord between its endpoints and a street cannot absorb
excess length by wiggling. Targets: chord_e = chord0_e * (c / v_e), i.e.
drawn length proportional to travel time, with the scale c chosen by least
squares (the typical street keeps its length).

Objective (relative stress + weak anchor for recognizability):
    E = sum_e ((|p_u - p_v| - d_e) / d_e)^2  +  w_a * sum_i |p_i - p0_i|^2 / S^2
"""

import argparse
import json
import pathlib
import time

import networkx as nx
import numpy as np
import osmnx as ox
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_pruned(path):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(DATA / "manhattan.graphml"))
    ap.add_argument("--out", default=str(DATA / "network2d.json"))
    ap.add_argument("--vref", type=float, default=0.0,
                    help="kph that keeps its length (0 = least-squares scale)")
    ap.add_argument("--anchor", type=float, default=0.05,
                    help="anchor weight (per node, vs unit relative stress)")
    ap.add_argument("--straight", type=float, default=3.0,
                    help="collinearity-through-intersections weight")
    ap.add_argument("--maxiter", type=int, default=5000)
    args = ap.parse_args()

    G = load_pruned(args.graph)
    print(f"{len(G.nodes)} nodes, {len(G.edges)} edges")

    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    P0 = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes])
    P0 -= P0.mean(0)
    N = len(P0)

    center = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in nodes]).mean(0)
    ea, eb, speed, geom = [], [], [], []
    for u, v, k, d in G.edges(keys=True, data=True):
        ea.append(idx[u]); eb.append(idx[v])
        speed.append(float(d.get("speed_kph", 40.0)))
        if "geometry" in d:
            geom.append(np.array(d["geometry"].coords) - center)
        else:
            geom.append(np.array([P0[idx[u]], P0[idx[v]]]))
    ea = np.array(ea); eb = np.array(eb); speed = np.array(speed)

    chord0 = np.sqrt(((P0[ea] - P0[eb]) ** 2).sum(1))
    ok = chord0 > 5.0          # drop self-loops/degenerate from the objective
    print(f"{(~ok).sum()} degenerate edges excluded from objective")

    # least-squares proportionality constant: chord target d = chord0 * c / v
    if args.vref > 0:
        c = args.vref
    else:
        r = chord0[ok] / speed[ok]
        c = (chord0[ok] * r).sum() / (r * r).sum()
    d = chord0 * c / speed
    print(f"scale c = {c:.1f} kph keeps its length; warp range "
          f"{(c/speed).min():.2f}..{(c/speed).max():.2f}")

    A, B = ea[ok], eb[ok]
    D = d[ok]
    span2 = ((P0.max(0) - P0.min(0)) ** 2).sum()
    wa = args.anchor / span2

    # collinearity through intersections: where a street continues nearly
    # straight through a node (original chord directions ~opposite), keep it
    # straight. Streets become stiff rods, so surplus length must surface as
    # neighborhood-scale waves instead of block-scale herringbone shear.
    incident = {}
    for i in range(len(ea)):
        if not ok[i]:
            continue
        for n, other in ((ea[i], eb[i]), (eb[i], ea[i])):
            u = (P0[other] - P0[n]) / max(np.linalg.norm(P0[other] - P0[n]), 1e-9)
            incident.setdefault(n, []).append((other, u))
    cn, c1, c2 = [], [], []
    for n, lst in incident.items():
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][1] @ lst[j][1] < -0.75 and lst[i][0] != lst[j][0]:
                    cn.append(n); c1.append(lst[i][0]); c2.append(lst[j][0])
    cn = np.array(cn); c1 = np.array(c1); c2 = np.array(c2)
    ws = args.straight
    print(f"{len(cn)} collinearity constraints")

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

        if len(cn):
            d1 = P[c1] - P[cn]
            d2 = P[c2] - P[cn]
            L1 = np.sqrt((d1 * d1).sum(1) + 1e-9)
            L2 = np.sqrt((d2 * d2).sum(1) + 1e-9)
            u1 = d1 / L1[:, None]
            u2 = d2 / L2[:, None]
            s = u1 + u2                      # 0 when exactly opposite
            E += ws * (s * s).sum()
            # dE/dd1 = 2 ws (I - u1 u1^T) s / L1
            g1 = 2.0 * ws * (s - (s * u1).sum(1)[:, None] * u1) / L1[:, None]
            g2 = 2.0 * ws * (s - (s * u2).sum(1)[:, None] * u2) / L2[:, None]
            np.add.at(grad, c1, g1)
            np.add.at(grad, c2, g2)
            np.add.at(grad, cn, -(g1 + g2))

        dxy = P - P0
        E += wa * (dxy * dxy).sum()
        grad += 2.0 * wa * dxy
        return E, grad.ravel()

    E0, _ = f(P0.ravel())
    t0 = time.time()
    res = minimize(f, P0.ravel().copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": args.maxiter, "maxfun": args.maxiter * 2})
    P = res.x.reshape(N, 2)
    L = np.sqrt(((P[A] - P[B]) ** 2).sum(1))
    rel = L / D
    print(f"{res.message.split(',')[0]} {time.time()-t0:.0f}s; "
          f"stress {E0:.0f} -> {res.fun:.0f}")
    print(f"realized/target quartiles: {np.percentile(rel,25):.2f} / "
          f"{np.percentile(rel,50):.2f} / {np.percentile(rel,75):.2f}")
    shift = np.sqrt(((P - P0) ** 2).sum(1))
    print(f"node shift mean {shift.mean():.0f}m max {shift.max():.0f}m")

    # carry interior geometry along with a per-edge similarity transform
    # (complex math: z' = a z + b mapping old endpoints onto new ones)
    out_edges = []
    for i in range(len(ea)):
        u, v = ea[i], eb[i]
        pts0 = geom[i]
        z0 = pts0[:, 0] + 1j * pts0[:, 1]
        a0, b0 = z0[0], z0[-1]
        a1 = P[u, 0] + 1j * P[u, 1]
        b1 = P[v, 0] + 1j * P[v, 1]
        if abs(b0 - a0) > 1e-9:
            s = (b1 - a1) / (b0 - a0)
            z1 = a1 + s * (z0 - a0)
        else:
            z1 = z0 + (a1 - a0)
        flat = np.column_stack([pts0[:, 0], pts0[:, 1]])
        pos = np.column_stack([z1.real, z1.imag, np.zeros(len(z1))])
        out_edges.append({
            "speed": round(float(speed[i]), 1),
            "hw": "", "warp": round(float(c / speed[i]), 3),
            "flat": np.round(flat, 1).ravel().tolist(),
            "pos": np.round(pos, 1).ravel().tolist(),
        })

    out = {
        "meta": {
            "mode": "2d-edge-stress", "scale_kph": float(c),
            "anchor": args.anchor, "n_points": N, "z_max": 0.0,
            "extent": [float(P0[:, 0].min()), float(P0[:, 0].max()),
                       float(P0[:, 1].min()), float(P0[:, 1].max())],
        },
        "edges": out_edges,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
