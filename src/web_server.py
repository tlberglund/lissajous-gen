"""Embedded HTTP server: the control panel's static files and JSON API.

Runs on a daemon thread. It never touches the renderer, generator, or
mixer; it validates input, updates AppState, and queues commands for the
main loop. A failure here must never take the display down.
"""

import json
import mimetypes
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from config_store import PresetError, PresetStore, ValidationError, validate_and_normalize
from control import AppState, ApplyConfig, Restart, Transport

WEB_ROOT = Path(__file__).resolve().parent / "web"
MUSIC_ACTIONS = ("play", "pause", "resume", "next", "prev")
MAX_BODY = 1 << 20

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("font/woff2", ".woff2")


class ApiError(Exception):
    def __init__(self, status: int, body: Dict[str, Any]):
        self.status = status
        self.body = body


def _make_handler(state: AppState, presets: PresetStore, web_root: Path):

    class Handler(BaseHTTPRequestHandler):
        server_version = "Lissajous/1.0"

        # -- plumbing -------------------------------------------------------

        def log_message(self, fmt, *args):
            pass    # keep the console for the visualizer's own output

        def _send_json(self, status: int, body: Any):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ApiError(413, {"error": "request body too large"})
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise ApiError(400, {"errors": [{"path": "", "message": f"invalid JSON: {e}"}]})

        def _dispatch(self, method: str):
            try:
                path = urlparse(self.path).path
                if path.startswith("/api/"):
                    self._api(method, [unquote(p) for p in path[5:].split("/") if p])
                elif method == "GET":
                    self._static(path)
                else:
                    raise ApiError(405, {"error": "method not allowed"})
            except ApiError as e:
                self._send_json(e.status, e.body)
            except Exception:
                print("ERROR handling web request:")
                traceback.print_exc()
                try:
                    self._send_json(500, {"error": "internal server error"})
                except Exception:
                    pass

        def do_GET(self):
            self._dispatch("GET")

        def do_PUT(self):
            self._dispatch("PUT")

        def do_POST(self):
            self._dispatch("POST")

        # -- static files ---------------------------------------------------

        def _static(self, path: str):
            rel = unquote(path).lstrip("/") or "index.html"
            root = web_root.resolve()
            target = (root / rel).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ApiError(404, {"error": "not found"})
            data = target.read_bytes()
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        # -- API ------------------------------------------------------------

        def _api(self, method: str, parts: list):
            route = (method, parts[0] if parts else "")
            if route == ("GET", "config") and len(parts) == 1:
                return self._send_json(200, self._config_body())
            if route == ("PUT", "config") and len(parts) == 1:
                return self._put_config(self._read_json())
            if route == ("GET", "presets") and len(parts) == 1:
                return self._send_json(200, {"presets": presets.list()})
            if route == ("POST", "presets") and len(parts) == 3 and parts[2] == "load":
                return self._load_preset(parts[1])
            if route == ("POST", "presets") and len(parts) == 2:
                return self._save_preset(parts[1], self._read_json())
            if route == ("GET", "music") and len(parts) == 1:
                return self._send_json(200, self._music_body())
            if route == ("POST", "music") and len(parts) == 2:
                return self._music_action(parts[1], self._read_json())
            if route == ("POST", "restart") and len(parts) == 1:
                state.commands.put(Restart())
                return self._send_json(202, {"restarting": True})
            if route == ("GET", "status") and len(parts) == 1:
                snap = state.snapshot()
                return self._send_json(200, {"source": snap.source,
                                             "fps": round(snap.fps, 1),
                                             "uptime": round(state.uptime(), 1)})
            raise ApiError(404, {"error": "not found"})

        def _config_body(self) -> Dict[str, Any]:
            return {"config": state.desired(), "restart_required": state.restart_required()}

        def _accept(self, doc: Dict[str, Any]):
            """Make a validated document the desired config and queue it."""
            state.set_desired(doc)
            state.commands.put(ApplyConfig(doc))

        def _put_config(self, body: Any):
            doc, errors = validate_and_normalize(body)
            if errors:
                raise ApiError(400, {"errors": errors})
            if doc["input_directory"] != state.boot_input_directory:
                raise ApiError(400, {"errors": [{
                    "path": "input_directory",
                    "message": "the music directory cannot be changed from the web interface"}]})
            self._accept(doc)
            self._send_json(200, {"restart_required": state.restart_required()})

        def _load_preset(self, name: str):
            try:
                doc = presets.load(name)
            except ValidationError as e:
                raise ApiError(400, {"errors": e.errors})
            except PresetError as e:
                raise ApiError(e.status, {"error": str(e)})
            # The music directory is fixed for the process, whatever the preset says
            doc["input_directory"] = state.boot_input_directory
            self._accept(doc)
            self._send_json(200, self._config_body())

        def _save_preset(self, name: str, body: Any):
            overwrite = bool(body.get("overwrite")) if isinstance(body, dict) else False
            try:
                presets.save(name, state.desired(), overwrite=overwrite)
            except PresetError as e:
                raise ApiError(e.status, {"error": str(e)})
            self._send_json(200, {"saved": name, "presets": presets.list()})

        def _music_body(self) -> Dict[str, Any]:
            snap = state.snapshot()
            return {"source": snap.source, "tracks": snap.tracks,
                    "current": snap.current_track, "playing": snap.playing}

        def _music_action(self, action: str, body: Any):
            if action not in MUSIC_ACTIONS:
                raise ApiError(404, {"error": f"unknown action {action!r}"})
            track = None
            if action == "play":
                track = body.get("track") if isinstance(body, dict) else None
                if track not in state.snapshot().tracks:
                    raise ApiError(404, {"error": f"unknown track {track!r}"})
            state.commands.put(Transport(action, track))
            self._send_json(202, {"queued": action})

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_web_server(state: AppState, presets: PresetStore, port: int,
                     host: str = "0.0.0.0", web_root: Path = WEB_ROOT) -> Optional[_Server]:
    """Start the server on a daemon thread. Returns None if it can't bind."""
    try:
        server = _Server((host, port), _make_handler(state, presets, web_root))
    except OSError as e:
        print(f"ERROR: web interface unavailable, could not listen on {host}:{port}: {e}")
        return None
    thread = threading.Thread(target=server.serve_forever, name="web-server", daemon=True)
    thread.start()
    shown = "localhost" if host in ("0.0.0.0", "") else host
    print(f"Web interface: http://{shown}:{server.server_address[1]}/")
    return server
