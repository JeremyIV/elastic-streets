"""Pilot part 4: decide the speed basis. The correct one must DECREASE as
congestion_numeric increases (0=free-flow, ~100=stopped). Compare distance/
duration vs the `speed` annotation by correlating each with numeric, within
the well-sampled classes. Free — operates on saved links.
"""
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent.parent
links = json.loads((ROOT / "data" / "pilot_nyc_links.json").read_text())
# reattach class from the matched run if present; else re-skip
mm = json.loads((ROOT / "data" / "pilot_nyc_links.json").read_text())

# pilot_classmatch wrote class onto the in-memory list but saved pilot_nyc_links
# without it; recompute class membership cheaply by reloading via the printed
# data is unavailable, so just use numeric-vs-speed pooled and per the few links
# that carry 'class'. Fallback: pool all, normalized per nothing.
def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = statistics.pstdev(xs)
    sy = statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)

rows = [l for l in links if l.get("numeric") is not None]
num = [l["numeric"] for l in rows]
dd = [l["kmh_distdur"] for l in rows]
an = [l["kmh_ann"] for l in rows]

print(f"{len(rows)} links with numeric")
print(f"corr(numeric, distance/duration speed) = {corr(num, dd):+.3f}")
print(f"corr(numeric, speed annotation)        = {corr(num, an):+.3f}")
print("(correct basis should be NEGATIVE: more congestion -> lower speed)")

# bin by numeric, show median speed each basis
print(f"\n{'numeric bin':12s} {'n':>3s} {'dd_kmh':>7s} {'ann_kmh':>7s}")
bins = [(0, 1), (1, 20), (20, 40), (40, 60), (60, 100)]
for lo, hi in bins:
    sel = [(l["kmh_distdur"], l["kmh_ann"]) for l in rows
           if lo <= l["numeric"] < hi]
    if sel:
        print(f"  [{lo:3d},{hi:3d})   {len(sel):3d} "
              f"{statistics.median(x for x,_ in sel):7.1f} "
              f"{statistics.median(y for _,y in sel):7.1f}")
