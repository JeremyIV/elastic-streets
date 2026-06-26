"""Tiny mobile-friendly server for the breathing videos, reachable over Tailscale.

Serves ONLY shots/*.mp4 (never the repo — keeps .mapbox_token etc. private),
with HTTP Range support so the clips also stream/scrub inline on a phone.
Auto-lists whatever videos exist, so new cities appear on refresh.

Usage: serve_videos.py [port]   (binds 0.0.0.0; reach via the Tailscale IP)
"""
import html
import os
import pathlib
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHOTS = ROOT / "shots"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8788


# clear labels + display order (new tile-method cities first, originals after)
LABELS = {
    "breathing_seattle_dir.mp4": ("Seattle — CONTINUOUS method (NEW \u2b50)", "Directions speeds \u00b7 \u221215%\u2026+23% \u00b7 rich!", -3),
    "snap_seattle_dir.png": ("Seattle — CONTINUOUS speed (NEW method)",
                             "4am / 8am / 5pm · 6%→40% slowed — rich!", 0),
    "congmap_nyc.png": ("Manhattan — tile category (old method)",
                        "sparse by comparison", 0),
    "diag_nyc.mp4": ("Manhattan — raw congestion (DIAGNOSTIC)",
                     "no warp · category colors · night→rush pulse", -2),
    "diag_seattle.mp4": ("Seattle — raw congestion (DIAGNOSTIC)",
                         "no warp · barely changes (peak only 9% congested)", -1),
    "breathing_nyc.mp4": ("New York", "live tiles · −11%…+16%", 0),
    "breathing_chicago.mp4": ("Chicago", "live tiles · −12%…+15%", 1),
    "breathing_sf.mp4": ("San Francisco", "live tiles · −10%…+18% (Bay barbell)", 2),
    "breathing_boston.mp4": ("Boston", "live tiles · −2%…+6% (faint)", 3),
    "breathing_nyc_v1_150lm.mp4": ("New York (v1 draft)", "150-landmark draft", 4),
    "breathing.mp4": ("Manhattan (original)", "2019 Uber", 10),
    "breathing_amp.mp4": ("Manhattan (amplified — the viral one)", "2019 Uber", 11),
    "breathing_planar.mp4": ("Manhattan (planar)", "2019 Uber", 12),
    "breathing_seattle.mp4": ("Seattle", "2019 Uber", 13),
    "breathing_seattle_amp.mp4": ("Seattle (amplified)", "2019 Uber", 14),
    "breathing_la.mp4": ("Los Angeles (anchor preview)", "Mapbox directions", 15),
    "breathing_la_full.mp4": ("Los Angeles", "Mapbox directions", 16),
}


def videos():
    return sorted(SHOTS.glob("*.mp4"),
                  key=lambda p: (LABELS.get(p.name, ("", "", 99))[2], p.name))


def pretty(name):
    if name in LABELS:
        return LABELS[name][0]
    s = re.sub(r"^breathing_", "", name[:-4]).replace("_", " ")
    return s.title().replace("Nyc", "NYC").replace("Sf", "SF").replace("La", "LA")


def sublabel(name):
    return LABELS.get(name, ("", "", 99))[1]


def human(n):
    return f"{n/1e6:.1f} MB"


def safe(name):
    """Resolve a requested name to a real shots/*.mp4|png or None (no traversal)."""
    if not (name.endswith(".mp4") or name.endswith(".png")) or \
            "/" in name or "\\" in name:
        return None
    p = (SHOTS / name).resolve()
    if p.parent == SHOTS.resolve() and p.is_file():
        return p
    return None


def snapshots():
    return sorted(SHOTS.glob("*.png"), key=lambda p: p.name)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Elastic Streets — videos</title>
<style>
 body{{margin:0;background:#05070c;color:#e8eef6;
   font-family:-apple-system,Helvetica,Arial,sans-serif;padding:18px}}
 h1{{font-size:20px;font-weight:600;margin:6px 0 4px}}
 .sub{{color:#8aa;font-size:13px;margin-bottom:18px}}
 .card{{background:#0c1118;border:1px solid #1c2636;border-radius:14px;
   padding:14px;margin:0 0 18px}}
 .row{{display:flex;justify-content:space-between;align-items:baseline}}
 .name{{font-size:17px;font-weight:600}}
 .meta{{color:#7d90a8;font-size:12px}}
 video{{width:100%;border-radius:10px;margin-top:10px;background:#000}}
 a.dl{{display:inline-block;margin-top:10px;padding:10px 16px;border-radius:10px;
   background:#e8814a;color:#05070c;text-decoration:none;font-weight:600;font-size:15px}}
</style></head><body>
<h1>Elastic Streets</h1>
<div class=sub>{n} videos · tap ▶ to preview, or the button to save</div>
{cards}
</body></html>"""

CARD = """<div class=card>
 <div class=row><span class=name>{title}</span><span class=meta>{size}</span></div>
 <div class=meta>{sub}</div>
 <video controls preload=metadata playsinline src="/v/{name}"></video>
 <a class=dl href="/dl/{name}">⬇ Download</a>
</div>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            return self.index()
        if path.startswith("/v/"):
            return self.send_video(path[3:], download=False)
        if path.startswith("/dl/"):
            return self.send_video(path[4:], download=True)
        self.send_error(404)

    def index(self):
        vids = videos()
        cards = "\n".join(
            CARD.format(title=html.escape(pretty(p.name)), size=human(p.stat().st_size),
                        sub=html.escape(sublabel(p.name)), name=html.escape(p.name))
            for p in vids)
        snaps = snapshots()
        if snaps:
            cards += '<h1 style="margin-top:28px">Diagnostic snapshots</h1>'
            for p in snaps:
                cards += (
                    f'<div class=card><div class=name>{html.escape(pretty(p.name))}</div>'
                    f'<div class=meta>{html.escape(sublabel(p.name))}</div>'
                    f'<img src="/v/{html.escape(p.name)}" '
                    f'style="width:100%;border-radius:10px;margin-top:8px"></div>')
        body = PAGE.format(n=len(vids), cards=cards).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_video(self, name, download):
        p = safe(name)
        if p is None:
            return self.send_error(404)
        size = p.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        partial = False
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                g1, g2 = m.group(1), m.group(2)
                if g1 == "" and g2:                 # suffix bytes=-N
                    start = max(0, size - int(g2))
                else:
                    start = int(g1)
                    end = int(g2) if g2 else size - 1
                end = min(end, size - 1)
                partial = start <= end
        length = end - start + 1 if partial else size
        ctype = "image/png" if p.name.endswith(".png") else "video/mp4"
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition",
                             f'attachment; filename="{p.name}"')
        self.end_headers()
        with open(p, "rb") as f:
            if partial:
                f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"serving {len(videos())} videos on 0.0.0.0:{PORT}", flush=True)
    srv.serve_forever()
