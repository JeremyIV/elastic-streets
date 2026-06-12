"""Collect 24 hourly snapshots of the Mapbox live traffic layer for a city.

Fetches the mapbox-traffic-v1 vector tiles covering a bbox once per hour and
archives the raw (gzipped) tile bytes — decoding and matching to the street
graph happen offline, so collection is tiny and robust. ~60-100 tiles per
snapshot; a full city-day is ~2k tile requests (Vector Tiles API free tier:
200k/month).

Usage: traffic_collect.py west south east north out_dir [hours] [zoom]
e.g.:  traffic_collect.py -122.85 45.42 -122.47 45.62 data/pdx_traffic 24 13
"""

import json
import math
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKEN = (ROOT / ".mapbox_token").read_text().strip()
W, S, E, N = (float(a) for a in sys.argv[1:5])
OUT = pathlib.Path(sys.argv[5])
HOURS = int(sys.argv[6]) if len(sys.argv) > 6 else 24
Z = int(sys.argv[7]) if len(sys.argv) > 7 else 13
OUT.mkdir(parents=True, exist_ok=True)

def tile_xy(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y = int((1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n)
    return x, y

x0, y1 = tile_xy(W, S, Z)
x1, y0 = tile_xy(E, N, Z)
tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
print(f"{len(tiles)} tiles per snapshot, {HOURS} snapshots "
      f"= {len(tiles) * HOURS} requests", flush=True)
json.dump({"bbox": [W, S, E, N], "zoom": Z,
           "tiles": [list(t) for t in tiles]}, open(OUT / "manifest.json", "w"))

def fetch(x, y):
    url = (f"https://api.mapbox.com/v4/mapbox.mapbox-traffic-v1/"
           f"{Z}/{x}/{y}.mvt?access_token={TOKEN}")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:          # empty tile
                return b""
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None

for snap in range(HOURS):
    stamp = time.strftime("%Y%m%dT%H%M")
    sdir = OUT / f"snap_{stamp}"
    sdir.mkdir(exist_ok=True)
    t0 = time.time()
    misses = 0
    for (x, y) in tiles:
        b = fetch(x, y)
        if b is None:
            misses += 1
            continue
        (sdir / f"{x}_{y}.mvt").write_bytes(b)
        time.sleep(0.1)
    print(f"[{stamp}] snapshot {snap + 1}/{HOURS}: "
          f"{len(tiles) - misses}/{len(tiles)} tiles "
          f"({time.time() - t0:.0f}s)", flush=True)
    if snap < HOURS - 1:
        # sleep to the top of the next hour
        now = time.time()
        time.sleep(3600 - (now % 3600) + 5)
print("collection complete")
