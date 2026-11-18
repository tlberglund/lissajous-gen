"""Hand-off between the web server thread and the render (main) thread.

The server never touches GL, audio, or the generator. It validates input,
updates the shared AppState, and puts a command on a queue. The main loop
drains that queue once per frame and performs the actual changes.
"""

import copy
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config_store import restart_required_changes


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@dataclass
class ApplyConfig:
    doc: Dict[str, Any]


@dataclass
class Transport:
    action: str                 # play | pause | resume | next | prev
    track: Optional[str] = None


@dataclass
class Restart:
    pass


def drain(q: "queue.Queue") -> List[Any]:
    """Take everything off the queue without blocking.

    Only the last ApplyConfig survives (each one is a complete document, so
    earlier ones are obsolete); other commands keep their order.
    """
    cmds = []
    while True:
        try:
            cmds.append(q.get_nowait())
        except queue.Empty:
            break
    last_apply = max((i for i, c in enumerate(cmds) if isinstance(c, ApplyConfig)), default=None)
    return [c for i, c in enumerate(cmds)
            if not isinstance(c, ApplyConfig) or i == last_apply]


# ---------------------------------------------------------------------------
# Applying a config change
# ---------------------------------------------------------------------------

def _changed(old: Dict[str, Any], new: Dict[str, Any], section: str, key: str) -> bool:
    return old.get(section, {}).get(key) != new.get(section, {}).get(key)


def apply_config(app, old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """Push the differences between two normalized docs into the running app.

    `app` must provide: renderer (with set_* methods), generator (with
    retarget), music (with auto_advance and audio.amplitude_scale), and
    switch_source(name). Returns the restart-required fields that changed.
    """
    r = app.renderer
    viz_setters = {
        "trace_color": r.set_trace_color,
        "trace_width": r.set_trace_width,
        "decay_rate": r.set_decay_rate,
        "glow_radius": r.set_glow_radius,
        "glow_intensity": r.set_glow_intensity,
        "beam_sharpness": r.set_beam_sharpness,
        "background_color": r.set_background_color,
    }
    for key, setter in viz_setters.items():
        if _changed(old, new, "visualization", key):
            value = new["visualization"][key]
            setter(tuple(value) if isinstance(value, list) else value)

    if _changed(old, new, "audio", "amplitude_scale"):
        app.music.audio.amplitude_scale = new["audio"]["amplitude_scale"]

    if _changed(old, new, "playback", "auto_advance"):
        app.music.auto_advance = new["playback"]["auto_advance"]

    old_sg, new_sg = old["signal_generator"], new["signal_generator"]
    if any(old_sg.get(k) != new_sg.get(k) for k in ("x", "y", "transition_time")):
        app.generator.retarget(new_sg)

    if old_sg.get("enabled") != new_sg.get("enabled"):
        app.switch_source("generator" if new_sg["enabled"] else "music")

    return restart_required_changes(old, new)


# ---------------------------------------------------------------------------
# Shared state read by the server
# ---------------------------------------------------------------------------

@dataclass
class _Snapshot:
    source: str = "generator"
    fps: float = 0.0
    tracks: List[str] = field(default_factory=list)
    current_track: Optional[str] = None
    playing: bool = False


class AppState:
    """Thread-safe state shared by the server (reader/writer) and main loop.

    desired: the config the UI should see (last accepted document).
    boot:    the config the process started with, for restart_required.
    """

    def __init__(self, boot_doc: Dict[str, Any]):
        self._lock = threading.Lock()
        self._boot = copy.deepcopy(boot_doc)
        self._desired = copy.deepcopy(boot_doc)
        self._snap = _Snapshot()
        self.started = time.monotonic()
        self.commands: "queue.Queue" = queue.Queue()

    @property
    def boot_input_directory(self) -> str:
        return self._boot["input_directory"]

    def desired(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._desired)

    def set_desired(self, doc: Dict[str, Any]):
        with self._lock:
            self._desired = copy.deepcopy(doc)

    def restart_required(self) -> bool:
        with self._lock:
            return bool(restart_required_changes(self._boot, self._desired))

    def update_snapshot(self, **fields):
        with self._lock:
            for k, v in fields.items():
                setattr(self._snap, k, copy.copy(v))

    def snapshot(self) -> _Snapshot:
        with self._lock:
            return copy.deepcopy(self._snap)

    def uptime(self) -> float:
        return time.monotonic() - self.started
