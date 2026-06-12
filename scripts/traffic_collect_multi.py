"""Collect hourly Mapbox live-traffic snapshots for many cities at once.

Stdlib-only (runs on a bare cluster python3). Each hour, fetches the
mapbox-traffic-v1 tiles covering every city's bbox and archives the raw
bytes under <out>/<city>/snap_<UTCstamp>/. Hour-of-local-day is recovered
offline from the stamp + each city's tz. One process handles all cities.

Usage: traffic_collect_multi.py cities.json out_dir token_file [snapshots]
"""

import json
import math
import pathlib
import sys
import time
import urllib.request

CITIES = json.load(open(sys.argv[1]))
OUT = pathlib.Path(sys.argv[2])
TOKEN = pathlib.Path(sys.argv[3]).read_text().strip()
SNAPS = int(sys.argv[4]) if len(sys.argv) > 4 else 26
Z = 13

def tile_xy(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    lr = math.radians(lat)
    y = int((1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n)
    return x, y

plans = []
total = 0
for c in CITIES:
    w, s, e, n = c["bbox"]
    x0, y1 = tile_xy(w, s, Z)
    x1, y0 = tile_xy(e, n, Z)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    plans.append((c["name"], tiles))
    d = OUT / c["name"]
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"bbox": c["bbox"], "zoom": Z, "tz": c["tz"],
               "tiles": [list(t) for t in tiles]},
              open(d / "manifest.json", "w"))
    total += len(tiles)
    print(f"{c['name']}: {len(tiles)} tiles", flush=True)
print(f"{total} tiles/snapshot x {SNAPS} = {total * SNAPS} requests", flush=True)

def fetch(x, y):
    url = ("https://api.mapbox.com/v4/mapbox.mapbox-traffic-v1/"
           "{}/{}/{}.mvt?access_token={}".format(Z, x, y, TOKEN))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None

for snap in range(SNAPS):
    stamp = time.strftime("%Y%m%dT%H%M", time.gmtime())
    t0 = time.time()
    for name, tiles in plans:
        sdir = OUT / name / ("snap_" + stamp)
        sdir.mkdir(exist_ok=True)
        ok = 0
        for (x, y) in tiles:
            b = fetch(x, y)
            if b:
                (sdir / "{}_{}.mvt".format(x, y)).write_bytes(b)
            if b is not None:
                ok += 1
            time.sleep(0.08)
        print("[{}] {} {}/{} tiles".format(stamp, name, ok, len(tiles)),
              flush=True)
    print("[{}] snapshot {}/{} done in {:.0f}s".format(
        stamp, snap + 1, SNAPS, time.time() - t0), flush=True)
    if snap < SNAPS - 1:
        now = time.time()
        time.sleep(max(3600 - (now % 3600) + 5, 60))
print("collection complete", flush=True)
