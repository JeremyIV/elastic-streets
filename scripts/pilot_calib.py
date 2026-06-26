"""Pilot: confirm Mapbox Directions returns per-segment congestion annotations.

De-risks the calibration design before spending ~2k API calls. One driving-traffic
request along a class-diverse NYC route at Monday 8am (depart_at), full annotations.
Reports which annotation arrays came back, their lengths, the congestion category
distribution, congestion_numeric range, and speed cross-check. Saves raw JSON.

Stdlib only.
"""
import json
import pathlib
import sys
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOKEN = (ROOT / ".mapbox_token").read_text().strip()

# Times Square -> JFK: surface streets + Midtown Tunnel + LIE + Van Wyck.
ORIG = (-73.9855, 40.7580)
DEST = (-73.7781, 40.6413)
DEPART = "2026-06-15T08:00"  # Mon 8am local, rush hour, future (today is Sat 6-13)

coords = f"{ORIG[0]},{ORIG[1]};{DEST[0]},{DEST[1]}"
params = {
    "access_token": TOKEN,
    "annotations": "distance,duration,speed,congestion,congestion_numeric,maxspeed",
    "overview": "full",
    "geometries": "geojson",
    "steps": "false",
    "depart_at": DEPART,
}
url = (f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/"
       f"{urllib.parse.quote(coords)}?{urllib.parse.urlencode(params)}")

print("Requesting driving-traffic, depart_at", DEPART, "...", flush=True)
try:
    with urllib.request.urlopen(url, timeout=30) as r:
        status = r.getcode()
        body = r.read().decode()
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:2000])
    sys.exit(1)

print("HTTP", status)
data = json.loads(body)
(ROOT / "data").mkdir(exist_ok=True)
out = ROOT / "data" / "pilot_nyc_route.json"
out.write_text(body)
print("saved raw ->", out, f"({len(body)} bytes)")
print("response code:", data.get("code"))

routes = data.get("routes", [])
if not routes:
    print("NO ROUTES. full response:")
    print(json.dumps(data, indent=2)[:2000])
    sys.exit(1)

rt = routes[0]
print(f"\nroute: {rt['distance']/1000:.1f} km, {rt['duration']/60:.1f} min, "
      f"{len(rt['legs'])} leg(s)")
geom = rt["geometry"]["coordinates"]
print(f"full geometry: {len(geom)} coordinates -> {len(geom)-1} segments expected")

ann = rt["legs"][0]["annotation"]
print("\n=== annotation arrays present ===")
for key in ["distance", "duration", "speed", "congestion",
            "congestion_numeric", "maxspeed"]:
    if key in ann and ann[key] is not None:
        arr = ann[key]
        print(f"  {key:20s} len={len(arr)}  sample={arr[:4]}")
    else:
        print(f"  {key:20s} MISSING")

cong = ann.get("congestion")
if cong:
    from collections import Counter
    counts = Counter(cong)
    print("\n=== congestion category distribution ===")
    for cat in ["low", "moderate", "heavy", "severe", "unknown"]:
        print(f"  {cat:10s} {counts.get(cat,0):4d}  "
              f"{100*counts.get(cat,0)/len(cong):5.1f}%")

cn = ann.get("congestion_numeric")
if cn:
    vals = [v for v in cn if v is not None]
    nnull = sum(1 for v in cn if v is None)
    if vals:
        print(f"\ncongestion_numeric: {len(vals)} non-null "
              f"(range {min(vals)}-{max(vals)}), {nnull} null")

# Speed cross-check: distance/duration vs reported speed annotation
dist, dur, spd = ann.get("distance"), ann.get("duration"), ann.get("speed")
if dist and dur and spd:
    print("\n=== speed cross-check (first 6 segments) ===")
    print(f"  {'class?':8s} {'dist_m':>7s} {'dur_s':>6s} "
          f"{'kmh_calc':>9s} {'kmh_ann':>8s} {'cong':>9s} {'numeric':>8s}")
    for i in range(min(6, len(dist))):
        kmh_calc = (dist[i] / dur[i] * 3.6) if dur[i] else 0
        kmh_ann = spd[i] * 3.6
        cn_i = cn[i] if cn else "-"
        print(f"  {'?':8s} {dist[i]:7.1f} {dur[i]:6.1f} "
              f"{kmh_calc:9.1f} {kmh_ann:8.1f} {cong[i]:>9s} {str(cn_i):>8s}")

print("\nPILOT CORE OK" if (cong and cn) else "\nPILOT INCOMPLETE — check above")
