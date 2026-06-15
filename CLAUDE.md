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

## Reproduce the videos (fast path, no data downloads)

Solved 24-hour layouts are committed as `data/day_mds*.json`. Each video is
two steps: an expensive frame render (~10–30 min, cached to `data/frames<suffix>/`)
and a fast overlay+encode (~30 s, output `shots/breathing<suffix>.mp4`).

```bash
# render_base.py  <day_json>  <suffix>  [zoom]
# compose_video.py <suffix> <title> <time_scale>
```

| city | render_base | compose_video |
|---|---|---|
| Manhattan | `data/day_mds.json` `""` | `""` `Manhattan` `1.37` |
| Manhattan (planar) | `data/day_mds_planar.json` `_planar` | `_planar` `Manhattan` `1.37` |
| Manhattan (amplified)* | `data/day_mds_amp.json` `_amp` | `_amp` `Manhattan` `1.77` |
| Seattle | `data/day_mds_seattle.json` `_seattle` `1.2` | `_seattle` `Seattle` `1.52` |
| Seattle (amplified)* | `data/day_mds_seattle_amp.json` `_seattle_amp` `1.2` | `_seattle_amp` `Seattle` `1.62` |
| Los Angeles** | `data/day_mds_la.json` `_la` | `_la` `"Los Angeles"` `1.0` |
| Seattle (directions)† | `COLOR_MODE=abs` `data/day_mds_seattle_dir.json` `_seattle_dir` | `_seattle_dir` `Seattle` `1.0` |

Example (Manhattan amplified, the flagship):

```bash
.venv/bin/python scripts/render_base.py data/day_mds_amp.json _amp
.venv/bin/python scripts/compose_video.py _amp Manhattan 1.77
open shots/breathing_amp.mp4
```

\* "amplified" = hour-dependent calibration so rush hour swells to match
real-world (Google-validated) congestion; these are the best-looking versions.
\*\* LA's layout json is large and gitignored; regenerate it first (~2 min,
no API token needed — the raw API pull is committed):
`.venv/bin/python scripts/anchor_layout.py` (or `anchor_layout_full.py` for
the per-intersection solve, ~1–2 h). LA's graph must exist too — see below.

† "directions" = continuous per-segment speeds harvested from the Mapbox
Directions API (much denser edge coverage than the live-traffic tile sampler).
See "Directions-API speeds" below. Render with `COLOR_MODE=abs`; the layout
json is committed but the raw `*_dir_links.json` harvest is gitignored.

The `time_scale` numbers divide the 10-minute yardstick: raw graph times run
fast vs. reality (no intersection penalties), validated against OSRM/Google.
1.0 for LA because Mapbox typical-traffic times are already calibrated.

## Re-solve the layouts from scratch (full path)

Needs the 2019 Uber Movement speed data (~120 MB) and OSM graphs:

```bash
sh scripts/fetch_data.sh                                   # speed matrices
.venv/bin/python scripts/fetch_graph.py                    # Manhattan graph
.venv/bin/python scripts/fetch_graph.py "Seattle, Washington, USA" seattle.graphml
```

Then per city (Manhattan shown; Seattle adds `--graph data/seattle.graphml
--data-dir data/seattle --out data/day_mds_seattle.json`):

```bash
.venv/bin/python scripts/animate_mds.py                    # 24 MDS solves, ~20–60 min
.venv/bin/python scripts/calibrate_yard.py data/day_mds.json data/manhattan.graphml data
.venv/bin/python scripts/amplify_breathing.py data/day_mds.json data/day_mds_amp.json 1.77 2.37
# Seattle amplify anchors: 1.62 2.26
```

`--planar` on animate_mds.py adds a foldover barrier (much slower: ~1 h
Manhattan, ~12 h Seattle — generally not worth it).

## Any other city (no Uber data needed)

`scripts/anchor_pull.py` samples 24 h of typical-traffic drive times from the
Mapbox Directions API over a sparse anchor grid (freeway-chained; ~30 k
requests ≈ a city, free tier is 100 k/month). Needs a Mapbox token in
`.mapbox_token` (gitignored). Then `anchor_layout_full.py` solves every
intersection using corridors as the speed sensor, and the render pipeline
runs unchanged with `time_scale 1.0`. The LA graph used by the committed pull:

```python
ox.graph_from_bbox(bbox=(-118.66, 33.70, -118.10, 34.32), network_type="drive",
    custom_filter='["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link"]')
# -> save as data/la_full.graphml, then add_edge_speeds
```

## Directions-API speeds — denser coverage, any city (the `_dir` variants)

A newer Mapbox path than anchor_pull, used for the `_dir` videos. Instead of
point-to-point anchor times, it harvests the *per-segment* speed annotations
the Directions API returns along each route — ~150–250 located (lon, lat, dist,
dur) samples per call — giving far denser edge coverage than the live-traffic
tiles. Needs `.mapbox_token`. Two steps, then the normal render:

```bash
# 1. harvest typical-traffic per-segment speeds, 24 h (resumable; ~routes×24 calls)
.venv/bin/python scripts/collect_directions.py seattle data/seattle_full.graphml
# 2. map onto edges, impute gaps, 24 MDS solves -> data/day_mds_seattle_dir.json
.venv/bin/python scripts/solve_directions.py seattle data/seattle_full.graphml
# 3. render (abs coloring) + compose (time_scale 1.0, like LA's calibrated times)
COLOR_MODE=abs .venv/bin/python scripts/render_base.py data/day_mds_seattle_dir.json _seattle_dir
.venv/bin/python scripts/compose_video.py _seattle_dir Seattle 1.0
```

`solve_directions.py` prints edge coverage % (well above the tile approach).
Run it on `data/manhattan.graphml` for a directions-based Manhattan directly
comparable to the original Uber map — same network, different speed source.

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
- Solves print per-hour `stress/pair` and `breath %` — breathing should span
  roughly −15%…+20% (raw Uber cities), −40%…+40% (LA typical-traffic).
- Speed-matrix joins key on OSM way ids carried in the graphml — re-fetching
  a graph years later may lose some matches as OSM evolves; coverage prints
  at solve time (Manhattan ~67%, Seattle ~24% as committed).
