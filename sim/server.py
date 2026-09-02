"""Local dashboard for the paper arb engine."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .engine import LiveEngine

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def make_handler(engine: LiveEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, content_type: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/state":
                self._json(engine.snapshot())
                return
            if path in {"/", "/index.html"}:
                self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
                return
            filename = path.lstrip("/")
            candidate = (WEB_DIR / filename).resolve()
            if WEB_DIR.resolve() in candidate.parents and candidate.is_file():
                types = {".css": "text/css", ".js": "application/javascript", ".svg": "image/svg+xml"}
                self._file(candidate, types.get(candidate.suffix, "application/octet-stream"))
                return
            self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/start":
                engine.start()
                self._json(engine.snapshot())
                return
            if path == "/api/stop":
                engine.stop()
                self._json(engine.snapshot())
                return
            if path == "/api/reset":
                engine.reset()
                self._json(engine.snapshot())
                return
            if path == "/api/cycle":
                try:
                    result = engine.run_cycle()
                    self._json({"ok": True, **result, "state": engine.snapshot()})
                except Exception as exc:  # noqa: BLE001
                    self._json({"ok": False, "error": str(exc)}, 500)
                return
            self._json({"error": "not found"}, 404)

    return Handler


def serve(engine: LiveEngine, host: str = "0.0.0.0", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    return server
