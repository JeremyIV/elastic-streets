# Elastic Streets — guide for Claude Code sessions

Travel-time maps of city street networks, animated over 24 hours so the city
"breathes" — swelling at rush hour, contracting at night. Euclidean distance
in each frame ≈ drive time between points. Read README.md for the concepts;
this file is the operational guide.

## Environment setup (once)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python osmnx scipy numpy matplotlib pillow pyproj
```

Also required: `ffmpeg` on PATH (`brew install ffmpeg` / `apt install ffmpeg`)
and a serif font (macOS Times New Roman, Liberation Serif, or DejaVu Serif —
compose_video.py auto-detects).

Always run scripts with `.venv/bin/python` from the repo root.

## THE CANONICAL PIPELINE — use this for every new city

This is the current best path and the **only** one to use for new animations.
Data source is **Mapbox Directions + Map Matching** (no Uber, no live-traffic
tiles, no anchor-grid — those are all legacy, see "Legacy paths" below and never
mix them in). Outputs are named `data/day_mds_<city>_gold.json` and
`shots/breathing_<city>_gold.mp4`. Needs a Mapbox token in `.mapbox_token`
(gitignored); free tier is 100 k requests/month and one city is ~5–40 k calls.

Six steps. Steps 1–4 are the slow API harvest (resumable — safe to re-run);
5 is the gold solver; 6 is render+compose.

```bash
CITY=portland                                  # must exist in data/cities.json
GRAPH=data/${CITY}_full.graphml

# 1. street graph (arterial+: motorway..tertiary, osmnx free-flow speeds attached)
.venv/bin/python scripts/fetch_city_graph.py $CITY            # -> $GRAPH

# 2. arterial per-segment speeds, 24 h typical-traffic (Directions annotations)
.venv/bin/python scripts/collect_directions.py $CITY $GRAPH   # -> data/${CITY}_dir_links.json

# 3. residential per-segment speeds (Map Matching covers streets Directions skips)
.venv/bin/python scripts/collect_matching.py $CITY $GRAPH     # -> data/${CITY}_match_links.json

# 4. DATA-PREP: map links onto edges, impute gaps, class-calibrate free-flow
#    (auto-merges _dir_links + _match_links). Writes the per-edge speed layout.
.venv/bin/python scripts/solve_directions.py $CITY $GRAPH     # -> data/day_mds_${CITY}_dir.json

# 5. GOLD SOLVER: importance-sampled all-pairs MDS, anchor 2.5 + shape 0.1 (THE solver)
.venv/bin/python scripts/solve_sampled.py \
    data/day_mds_${CITY}_dir.json data/day_mds_${CITY}_gold.json --anchor 2.5 --shape 0.1

# 6. render frames (~10–30 min, cached to data/frames_<city>_gold/) then compose
.venv/bin/python scripts/render_base.py data/day_mds_${CITY}_gold.json _${CITY}_gold
.venv/bin/python scripts/compose_video.py _${CITY}_gold "Portland" 1.2
open shots/breathing_${CITY}_gold.mp4
```

`scripts/overnight_gold.sh` / `overnight_harvest.sh` run this loop over many
cities unattended (resilient + resumable) — copy their structure for batches.

### Why each step is what it is (don't regress these)

- **Data source = Directions + Map Matching, nothing else.** The goal is to
  cover cities Uber never mapped, so every speed must be Mapbox-derived. The two
  APIs are complementary: Directions only ever returns the *fastest* path (rides
  arterials, ~40% edge coverage alone); Map Matching snaps *chosen* traces onto
  the network, filling the residential streets Directions skips (lifts coverage
  to ~80%+). The retired tile sampler left ~60% of edges at a constant free-flow
  speed, so those cities barely congested — the symptom that motivated this path.
  All harvests use `depart_at` TYPICAL traffic for an upcoming weekday (a
  prediction, queryable now — no live polling), matching Uber's weekday-average.
- **Imputation (`solve_directions.py`).** Uncovered edge-hours are filled by IDW
  of the *congestion ratio* (speed/free-flow) of the 8 nearest covered edges,
  not raw speed — so a side street inherits its neighborhood's rush-hour shape.
  Uncovered-edge **free-flow** is set to the observed per-highway-class median
  (≥8 covered edges to calibrate), because posted speed limits overestimate real
  residential free-flow by ~70%; classes never observed fall back to osmnx
  speed_kph. Coverage % prints at solve time.
- **Gold solver = `solve_sampled.py --anchor 2.5 --shape 0.1`.** Importance-sampled
  (Horvitz–Thompson) all-pairs MDS: a spring between sampled node pairs, rest
  length = c·travel-time. Defaults `--graphlocal --wq 1 --ew 100 --hops 2` build
  a 2-hop street mesh for the local band + distance-sampled long pairs for gross
  structure — this is what killed the old "furry"/spiky-edge artifact (street
  edges are now constrained, unlike the legacy landmark solver). `--anchor 2.5`
  is a *small* similarity-invariant Procrustes pull toward the geographic shape
  (the chosen gold value): just enough to stabilize gauge without flattening the
  warp. The anchor scale is normalized by spring energy, so it is NOT comparable
  to solve_directions' old 0.05 — on this solver 0 = fully warped, ~30 = mild,
  ~200 = "looks like the normal geographic map". `--shape 0.1` is the
  as-similar-as-possible (ASAP) angle term: it penalizes each junction's street
  fan deviating from a rotation+scale of its geographic shape, killing the
  street-to-street *jitter* (incoherent local shear) while leaving local scale
  free so the breathing survives. Chicago sweep: jitter (Laplacian roughness)
  falls 0.685→0.52 and plateaus by ~0.1, breath stays ~62% of the no-shape 64%;
  ≥2 flattens the swell, ≤0.01 under-smooths. NB the jitter is the LOCAL mesh
  fitting a non-Euclidean metric, not the far field — weakening far springs
  (`--fw`) does nothing, and capping `--maxiter` under-converges the ASAP term
  (it relaxes slowly); to sweep it fast, `--init <prior_layout> --passes 1`
  warm-starts + run cities in parallel (quality-neutral; maxiter-capping is not).
- **Render straightens loop roads (`render_base.py`).** Each edge is drawn as
  its real street polyline, rotated+scaled so its endpoints land on the solved
  nodes (factor `(b-a)/d0`). Streets that loop back near their start have a tiny
  geographic chord `d0`, so that scale explodes and the curve "spools out" into a
  giant arc. The fix: any edge with sinuosity (arc/chord) > `SINU_MAX` (env,
  default 5) is drawn as a straight chord instead. Only ~0.1% of edges (real
  ramps/cul-de-sacs) are affected.
- **`time_scale 1.2` in compose.** Graph shortest-path times run ~20% fast vs
  reality (no intersection/signal penalties) — uniformly across hours, NOT a
  rush effect, so the swell stays faithful. `scripts/validate_warp.py` scatters
  layout distance vs live Mapbox `depart_at` times to confirm/recalibrate per
  city (it found api/map ≈ 1.18 → 1.2). Coloring: default `ratio` mode (speed vs
  each edge's daily max) is the gold look — do not pass `COLOR_MODE=abs` for new
  cities (that was a workaround for sparse categorical tile data).

### Adding a city not in cities.json

Append `{"name": ..., "tz": ..., "bbox": [w, s, e, n]}` to `data/cities.json`
(bbox = lon/lat of the metro window you want framed), then run from step 1.

### Solver tuning knobs (rarely needed; gold defaults are right)

`solve_sampled.py`: `--wq` stress taper (0 = 1/D² local/fewest-spikes, 1 = 1/D
Sammon sweet spot **default**, 2 = unweighted/global/spiky); `--ew` street-spring
over-weight (default 100); `--hops` local-mesh radius (default 2; denser = cleaner
fine structure); `--anchor` geographic pull (gold 2.5); `--lcut` local/sampled
cutoff; `--seed`. NB the `tt>1s` floor in `solve_hour` is load-bearing — a `>30s`
floor silently drops ~2/3 of short street springs. Diagnostic:
`HIGHLIGHT_STRETCH=8 .venv/bin/python scripts/render_base.py …` paints edges
stretched >8× their real length red. Takeaways: spikes are a *weighting* effect
(mild 1/D taper minimizes them); folds are inherent to flattening a non-Euclidean
metric into 2D; `--planar` removes folds but freezes the breathing (planar XOR
breathing).

## Legacy paths (do NOT use for new cities — kept only to reproduce old videos)

These predate the canonical pipeline and use retired data sources. Listed so you
recognize old artifacts; never build a new animation with them.

- **Uber Movement** (`animate_mds.py`, `calibrate_yard.py`, `amplify_breathing.py`,
  `fetch_data.sh`, `fetch_graph.py`) — 2019 Uber speed matrices; produced the
  original `data/day_mds.json` / `_amp` / `_seattle` / `_planar` flagships
  (`time_scale` 1.37–1.77). Coverage decays as OSM way-ids drift.
- **Live-traffic tiles** (`solve_tiles.py`, `data/day_mds_{boston,chicago,nyc,sf}.json`,
  mode `day-mds-tiles`) — sparse edge coverage (~40%), weak congestion signal.
  Superseded; re-harvest these cities via the canonical pipeline.
- **Anchor grid** (`anchor_pull.py`, `anchor_layout.py`, `anchor_layout_full.py`,
  `data/day_mds_la*.json`, mode `day-mds-anchor`) — point-to-point typical-traffic
  over a sparse grid; LA's original source. Superseded by Directions+Matching.
- **Landmark solver** = `solve_directions.py`'s *own* MDS output. In the canonical
  pipeline solve_directions is used ONLY as data-prep (step 4); its landmark
  layout (springs from a few landmarks, edges absent from the objective → furry)
  is the legacy solver that `solve_sampled.py` replaced.

## Gotchas

- render_base.py is the expensive layer; compose_video.py is the cheap one.
  Iterate on titles/layout/yardstick via compose only — never re-render frames
  for overlay tweaks. The overlay LAYOUT dict (pixels) is at the top of
  compose_video.py.
- Framing wobble: keep `ax.set_aspect("equal", adjustable="box")` in
  render_base.py — `"datalim"` causes per-frame reframing flicker.
- validate_times*.py expect to run from `scripts/` (they use `../data` paths).
- The viewer (`viewer/day.html?data=...`) needs `python3 -m http.server` from
  the repo root; it splines the same jsons interactively.
- Solves print per-hour `breath %` — for canonical Directions cities expect
  roughly −15%…+20% (dense grids like Manhattan/Seattle) up to ≈ ±30% (sprawl
  like LA). A city that barely breathes (~±2%) means poor edge coverage —
  re-check the coverage % from step 4 before rendering.
- `solve_directions.py` prints edge coverage % (covered vs imputed). Directions
  alone is ~40%; with Map Matching merged it should reach ~80%+. Low coverage =
  weak congestion signal (the retired tile cities sat at ~40% and looked dead).
- `solve_sampled.py` runs an all-pairs Dijkstra ×24; memory ≈ n_nodes²×8 bytes
  per hour (~3 GB at ~19 k nodes). For very large metros, run cities sequentially,
  not concurrently.
