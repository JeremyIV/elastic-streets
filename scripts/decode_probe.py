"""Sanity-check decode of one real collected NYC traffic tile: confirm gzip +
MVT structure, property names (class/congestion), geometry extent, and the
tile->lonlat slippy math against the tile's known bbox.
"""
import gzip
import math
import pathlib
from collections import Counter
import mapbox_vector_tile as mvt

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPDIR = sorted((ROOT / "data" / "traffic_snaps" / "nyc").glob("snap_*"))[0]
tiles = sorted(SNAPDIR.glob("*.mvt"))
print(f"snapshot {SNAPDIR.name}: {len(tiles)} tiles")

Z = 13
EXTENT = 4096


def tile_bounds(x, y, z):
    n = 2 ** z
    def lon(xt): return xt / n * 360 - 180
    def lat(yt):
        r = math.atan(math.sinh(math.pi * (1 - 2 * yt / n)))
        return math.degrees(r)
    return lon(x), lat(y + 1), lon(x + 1), lat(y)  # w,s,e,n


# pick the tile with the most traffic features
best = None
for t in tiles:
    x, y = map(int, t.stem.split("_"))
    raw = t.read_bytes()
    try:
        data = mvt.decode(gzip.decompress(raw))
    except Exception:
        data = mvt.decode(raw)
    layers = list(data.keys())
    feats = data.get("traffic", {}).get("features", [])
    if best is None or len(feats) > best[1]:
        best = (t, len(feats), layers, data)

t, nf, layers, data = best
x, y = map(int, t.stem.split("_"))
w, s, e, n = tile_bounds(x, y, Z)
print(f"\nrichest tile {t.name}: layers={layers}")
print(f"traffic features: {nf}")
print(f"tile bbox: ({w:.4f},{s:.4f})-({e:.4f},{n:.4f})  "
      f"~{(e-w)*111*math.cos(math.radians(n)):.1f}km wide")

feats = data["traffic"]["features"]
print("\nsample feature props:")
for f in feats[:5]:
    print("  ", f["properties"], "geomtype", f["geometry"]["type"],
          "ncoords", len(f["geometry"]["coordinates"]))

print("\nclass distribution:",
      dict(Counter(f["properties"].get("class") for f in feats)))
print("congestion distribution:",
      dict(Counter(f["properties"].get("congestion") for f in feats)))

# check coord range (should be 0..4096 tile-local, or already lon/lat?)
c0 = feats[0]["geometry"]["coordinates"]
flat = c0[0] if isinstance(c0[0][0], (list, tuple)) else c0
print("\nfirst geom coords sample:", flat[:3])
xs = [p[0] for p in flat]
ys = [p[1] for p in flat]
print(f"coord ranges x[{min(xs)},{max(xs)}] y[{min(ys)},{max(ys)}] "
      f"(EXTENT={EXTENT} => tile-local)")

# convert one tile-local point to lon/lat and check it lands in tile bbox
def to_lonlat(px, py):
    lon = w + (px / EXTENT) * (e - w)
    lat = s + (py / EXTENT) * (n - s)
    return lon, lat
lon, lat = to_lonlat(xs[0], ys[0])
print(f"first point -> lonlat ({lon:.4f},{lat:.4f})  "
      f"in bbox: {w<=lon<=e and s<=lat<=n}")
