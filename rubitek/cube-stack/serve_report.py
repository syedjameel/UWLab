"""Serve the report over the tailnet, with caching off.

WHY A CUSTOM HANDLER. `python -m http.server` sends Last-Modified and no cache directives, so a
browser that has seen the page once will re-serve index.html and data.json from its own cache. This
report is regenerated continuously -- new evaluations, new clips, new training curves -- and a
viewer reloading to check on it would silently get an old copy. That failure is invisible: the page
renders perfectly, with yesterday's numbers.

Bound to the Tailscale address rather than 0.0.0.0 on purpose: 0.0.0.0 would also publish it to
whatever LAN or cafe wifi this laptop is attached to. On the tailnet interface the only reachable
audience is this tailnet's own devices.
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import subprocess
import sys
from pathlib import Path

DIR = Path("/home/mattephi/Work/Rubitek/cube-stack/report")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8899


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def guess_type(self, path):
        t = super().guess_type(path)
        # Without an explicit charset the browser guesses latin-1 and every em-dash in the page
        # renders as mojibake.
        if t in ("text/html", "text/plain", "application/javascript", "text/css"):
            return t + "; charset=utf-8"
        if str(path).endswith(".json"):
            return "application/json; charset=utf-8"
        return t

    def log_message(self, fmt, *a):  # one line per request is noise on a page that polls
        pass


class Server(socketserver.ThreadingTCPServer):
    # Video seeking opens several connections at once; the single-threaded default serialises them
    # and makes scrubbing feel broken.
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    ip = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True).stdout.split()
    if not ip:
        raise SystemExit("tailscale is not up -- no address to bind to")
    addr = ip[0]
    import os

    os.chdir(DIR)
    with Server((addr, PORT), functools.partial(NoCache, directory=str(DIR))) as httpd:
        print(f"serving {DIR} on http://{addr}:{PORT}/", flush=True)
        httpd.serve_forever()
