import copy
import http.client
import json
import socket
from pathlib import Path

import pytest

from config_store import DEFAULTS, PresetStore
from control import AppState, ApplyConfig, Restart, Transport, drain
from web_server import start_web_server

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config"


@pytest.fixture
def env(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>panel</title>")
    (web / "panel.js").write_text("export {}")
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("config-generator", "config-music"):
        (cfg / f"{name}.json").write_text((REPO_CONFIG / f"{name}.json").read_text())
    (cfg / "active.json").write_text(json.dumps(DEFAULTS))
    (tmp_path / "secret.txt").write_text("nope")

    state = AppState(DEFAULTS)
    state.update_snapshot(tracks=["a.mp3", "b.mp3"], source="generator", fps=30.0)
    server = start_web_server(state, PresetStore(cfg), 0, host="127.0.0.1", web_root=web)
    port = server.server_address[1]

    def request(method, path, body=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        data = raw if raw is not None else (json.dumps(body) if body is not None else None)
        conn.request(method, path, body=data,
                     headers={"Content-Type": "application/json"} if data else {})
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        ctype = resp.getheader("Content-Type", "")
        return resp.status, (json.loads(payload) if "json" in ctype else payload)

    yield state, request, cfg, port
    server.shutdown()
    server.server_close()


def doc_with(**changes):
    d = copy.deepcopy(DEFAULTS)
    for path, value in changes.items():
        section, key = path.split("__")
        d[section][key] = value
    return d


# -- 6.1 server robustness --------------------------------------------------

def test_bind_failure_is_not_fatal(capsys):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    try:
        assert start_web_server(AppState(DEFAULTS), PresetStore(Path(".")), port,
                                host="127.0.0.1") is None
        assert "could not listen" in capsys.readouterr().out
    finally:
        blocker.close()


def test_handler_exception_returns_500(env, monkeypatch, capsys):
    state, request, _, _ = env
    monkeypatch.setattr(state, "uptime", lambda: 1 / 0)
    status, body = request("GET", "/api/status")
    assert status == 500
    captured = capsys.readouterr()
    assert "ZeroDivisionError" in captured.out + captured.err
    # The server keeps serving
    assert request("GET", "/api/presets")[0] == 200


# -- 6.2 static files -------------------------------------------------------

def test_root_serves_index(env):
    _, request, _, _ = env
    status, body = request("GET", "/")
    assert status == 200 and b"panel" in body


def test_path_traversal_blocked(env):
    _, _, _, port = env
    for path in ("/../config/active.json", "/../secret.txt", "/%2e%2e/secret.txt"):
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(f"GET {path} HTTP/1.0\r\n\r\n".encode())
        reply = s.recv(4096).decode()
        s.close()
        assert reply.startswith("HTTP/1.0 404"), path


# -- 6.3 config -------------------------------------------------------------

def test_get_config(env):
    _, request, _, _ = env
    status, body = request("GET", "/api/config")
    assert status == 200
    assert body == {"config": DEFAULTS, "restart_required": False}


def test_put_config_success(env):
    state, request, _, _ = env
    new = doc_with(visualization__glow_intensity=3.0)
    status, body = request("PUT", "/api/config", new)
    assert status == 200 and body == {"restart_required": False}
    assert request("GET", "/api/config")[1]["config"]["visualization"]["glow_intensity"] == 3.0
    cmds = drain(state.commands)
    assert len(cmds) == 1 and isinstance(cmds[0], ApplyConfig)


def test_put_restart_required(env):
    _, request, _, _ = env
    status, body = request("PUT", "/api/config", doc_with(display__fullscreen=True))
    assert status == 200 and body == {"restart_required": True}


def test_put_validation_failure(env):
    state, request, _, _ = env
    bad = copy.deepcopy(DEFAULTS)
    bad["signal_generator"]["x"]["keyframes"][0]["amplitude"] = 1.5
    status, body = request("PUT", "/api/config", bad)
    assert status == 400
    assert body["errors"][0]["path"] == "signal_generator.x.keyframes[0].amplitude"
    assert request("GET", "/api/config")[1]["config"] == DEFAULTS
    assert drain(state.commands) == []


def test_put_malformed_json(env):
    state, request, _, _ = env
    status, body = request("PUT", "/api/config", raw="{not json")
    assert status == 400 and "errors" in body
    assert drain(state.commands) == []


def test_put_cannot_change_music_directory(env):
    _, request, _, _ = env
    new = copy.deepcopy(DEFAULTS)
    new["input_directory"] = "/elsewhere"
    status, body = request("PUT", "/api/config", new)
    assert status == 400 and body["errors"][0]["path"] == "input_directory"


# -- 6.4 presets ------------------------------------------------------------

def test_list_presets(env):
    _, request, _, _ = env
    assert request("GET", "/api/presets") == (200, {"presets": ["config-generator", "config-music"]})


def test_load_preset(env):
    state, request, _, _ = env
    status, body = request("POST", "/api/presets/config-generator/load")
    assert status == 200
    assert body["config"]["signal_generator"]["enabled"] is True
    assert body["restart_required"] is True        # the preset is fullscreen
    assert isinstance(drain(state.commands)[0], ApplyConfig)


def test_load_missing_preset(env):
    _, request, _, _ = env
    assert request("POST", "/api/presets/nope/load")[0] == 404


def test_save_preset_and_conflict(env):
    _, request, cfg, _ = env
    status, body = request("POST", "/api/presets/spirals", {})
    assert status == 200 and "spirals" in body["presets"]
    assert (cfg / "spirals.json").exists()
    assert request("POST", "/api/presets/spirals", {})[0] == 409
    assert request("POST", "/api/presets/spirals", {"overwrite": True})[0] == 200


def test_save_preset_bad_name(env):
    _, request, _, _ = env
    assert request("POST", "/api/presets/active", {})[0] == 400


# -- 6.5 music and status ---------------------------------------------------

def test_music_listing(env):
    _, request, _, _ = env
    status, body = request("GET", "/api/music")
    assert status == 200
    assert body == {"source": "generator", "tracks": ["a.mp3", "b.mp3"],
                    "current": None, "playing": False}


def test_music_play_and_unknown_track(env):
    state, request, _, _ = env
    assert request("POST", "/api/music/play", {"track": "b.mp3"})[0] == 202
    assert drain(state.commands) == [Transport("play", "b.mp3")]
    assert request("POST", "/api/music/play", {"track": "zzz.mp3"})[0] == 404
    assert request("POST", "/api/music/rewind", {})[0] == 404
    assert drain(state.commands) == []


def test_music_transport_actions(env):
    state, request, _, _ = env
    for action in ("pause", "resume", "next", "prev"):
        assert request("POST", f"/api/music/{action}")[0] == 202
    assert [c.action for c in drain(state.commands)] == ["pause", "resume", "next", "prev"]


def test_status(env):
    _, request, _, _ = env
    status, body = request("GET", "/api/status")
    assert status == 200 and body["source"] == "generator" and body["fps"] > 0


def test_restart_queues_command(env):
    state, request, _, _ = env
    assert request("POST", "/api/restart")[0] == 202
    assert drain(state.commands) == [Restart()]


def test_unknown_api_route(env):
    _, request, _, _ = env
    assert request("GET", "/api/nope")[0] == 404
