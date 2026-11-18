"""Configuration store: validation, normalization, persistence, and presets.

The canonical config is a plain JSON-compatible dict with every section
filled in (see DEFAULTS). All writers -- startup, the web API, preset
load/save -- go through validate_and_normalize() so there is exactly one
definition of what a valid config is.
"""

import copy
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

WAVEFORMS = ("sine", "triangle", "sawtooth", "square")
PLAYBACK_MODES = ("sequential", "random", "single-loop")
PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
RAND_RE = re.compile(r"^rand\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)$")
ACTIVE_NAME = "active"

DEFAULTS: Dict[str, Any] = {
    "input_directory": "./music",
    "display": {
        "resolution": "auto",
        "fullscreen": False,
    },
    "visualization": {
        "trace_color": [0.2, 1.0, 0.3],
        "trace_width": 2.0,
        "persistence_time": 1.5,
        "decay_rate": 0.7,
        "glow_radius": 8.0,
        "glow_intensity": 1.8,
        "beam_sharpness": 3.0,
        "background_color": [0.0, 0.0, 0.0],
    },
    "audio": {
        "sample_rate": 44100,
        "buffer_size": 2048,
        "amplitude_scale": 0.9,
    },
    "playback": {
        "mode": "sequential",
        "auto_advance": True,
    },
    "signal_generator": {
        "enabled": False,
        "transition_time": 1.5,
        "x": {
            "waveform": "sine",
            "keyframes": [{"frequency": 3.0, "phase": 0.0, "amplitude": 0.8, "duration": 1.0}],
        },
        "y": {
            "waveform": "sine",
            "keyframes": [{"frequency": 4.0, "phase": 0.0, "amplitude": 0.8, "duration": 1.0}],
        },
    },
    "web": {
        "port": 8080,
    },
}

# Settings that only take effect after a restart (dotted section.key paths).
RESTART_REQUIRED_FIELDS = (
    "display.resolution",
    "display.fullscreen",
    "audio.sample_rate",
    "audio.buffer_size",
    "web.port",
)

# (min, max) ranges for numeric fields; None means unbounded on that side.
_VIS_RANGES = {
    "trace_width": (0.1, 50.0),
    "persistence_time": (0.0, 60.0),
    "decay_rate": (0.0, 20.0),
    "glow_radius": (0.0, 50.0),
    "glow_intensity": (0.0, 10.0),
    "beam_sharpness": (0.5, 10.0),
}
_KF_RANGES = {
    "frequency": (0.0, 20000.0),   # exclusive lower bound, see _check_kf_number
    "phase": (None, None),
    "amplitude": (0.0, 1.0),
    "duration": (0.0, 3600.0),     # exclusive lower bound
}
_KF_EXCLUSIVE_MIN = ("frequency", "duration")


class ValidationError(Exception):
    """Raised when a document fails validation; carries the error list."""

    def __init__(self, errors: List[Dict[str, str]]):
        self.errors = errors
        super().__init__("; ".join(f"{e['path']}: {e['message']}" for e in errors))


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _fmt_num(v: float) -> str:
    """Format a number compactly for rand() strings (3.0 -> '3', 1.57 -> '1.57')."""
    if float(v).is_integer():
        return str(int(v))
    return repr(float(v))


class _Validator:
    def __init__(self):
        self.errors: List[Dict[str, str]] = []

    def err(self, path: str, message: str):
        self.errors.append({"path": path, "message": message})

    # -- primitives ---------------------------------------------------------

    def number(self, value: Any, path: str, lo=None, hi=None,
               exclusive_lo: bool = False, integer: bool = False) -> Optional[float]:
        """Coerce numeric strings, check range. Returns the number or None on error."""
        if isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                self.err(path, f"must be a number, got {value!r}")
                return None
            if not math.isfinite(value):
                self.err(path, "must be a finite number")
                return None
        if not _is_number(value):
            self.err(path, f"must be a number, got {type(value).__name__}")
            return None
        if integer:
            if float(value) != int(value):
                self.err(path, "must be a whole number")
                return None
            value = int(value)
        else:
            value = float(value)
        if lo is not None:
            if exclusive_lo and value <= lo:
                self.err(path, f"must be greater than {_fmt_num(lo)}")
                return None
            if not exclusive_lo and value < lo:
                self.err(path, f"must be at least {_fmt_num(lo)}")
                return None
        if hi is not None and value > hi:
            self.err(path, f"must be at most {_fmt_num(hi)}")
            return None
        return value

    def boolean(self, value: Any, path: str) -> Optional[bool]:
        if not isinstance(value, bool):
            self.err(path, "must be true or false")
            return None
        return value

    def color(self, value: Any, path: str) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            self.err(path, "must be a list of three numbers [r, g, b]")
            return None
        out = []
        for i, c in enumerate(value):
            n = self.number(c, f"{path}[{i}]", 0.0, 1.0)
            if n is None:
                return None
            out.append(n)
        return out

    def section(self, doc: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = doc.get(key, {})
        if not isinstance(value, dict):
            self.err(key, "must be an object")
            return {}
        return value

    # -- keyframe values ----------------------------------------------------

    def kf_value(self, value: Any, path: str, field: str) -> Any:
        """Validate a keyframe value: a number or 'rand(min, max)'.

        Returns a float, a canonical 'rand(a, b)' string, or None on error.
        """
        lo, hi = _KF_RANGES[field]
        excl = field in _KF_EXCLUSIVE_MIN
        if isinstance(value, str) and value.strip().lower().startswith("rand"):
            m = RAND_RE.match(value.strip())
            if not m:
                self.err(path, f"malformed rand expression {value!r}; expected rand(min, max)")
                return None
            before = len(self.errors)
            a = self.number(m.group(1), path, lo, hi, excl)
            b = self.number(m.group(2), path, lo, hi, excl)
            if len(self.errors) > before:
                return None
            if a > b:
                a, b = b, a
            return f"rand({_fmt_num(a)}, {_fmt_num(b)})"
        return self.number(value, path, lo, hi, excl)

    # -- sections -----------------------------------------------------------

    def display(self, src: Dict[str, Any], out: Dict[str, Any]):
        if "resolution" in src:
            r = src["resolution"]
            path = "display.resolution"
            if r == "auto":
                out["resolution"] = "auto"
            elif isinstance(r, str) and re.match(r"^\s*\d+\s*x\s*\d+\s*$", r):
                w, h = (int(p) for p in r.lower().split("x"))
                if w < 64 or h < 64 or w > 8192 or h > 8192:
                    self.err(path, "width and height must be between 64 and 8192")
                else:
                    out["resolution"] = [w, h]
            elif isinstance(r, (list, tuple)) and len(r) == 2:
                w = self.number(r[0], f"{path}[0]", 64, 8192, integer=True)
                h = self.number(r[1], f"{path}[1]", 64, 8192, integer=True)
                if w is not None and h is not None:
                    out["resolution"] = [w, h]
            else:
                self.err(path, 'must be "auto", "WIDTHxHEIGHT", or [width, height]')
        if "fullscreen" in src:
            b = self.boolean(src["fullscreen"], "display.fullscreen")
            if b is not None:
                out["fullscreen"] = b

    def visualization(self, src: Dict[str, Any], out: Dict[str, Any]):
        for key in ("trace_color", "background_color"):
            if key in src:
                c = self.color(src[key], f"visualization.{key}")
                if c is not None:
                    out[key] = c
        for key, (lo, hi) in _VIS_RANGES.items():
            if key in src:
                n = self.number(src[key], f"visualization.{key}", lo, hi)
                if n is not None:
                    out[key] = n

    def audio(self, src: Dict[str, Any], out: Dict[str, Any]):
        if "sample_rate" in src:
            n = self.number(src["sample_rate"], "audio.sample_rate", 8000, 192000, integer=True)
            if n is not None:
                out["sample_rate"] = n
        if "buffer_size" in src:
            n = self.number(src["buffer_size"], "audio.buffer_size", 64, 16384, integer=True)
            if n is not None:
                out["buffer_size"] = n
        if "amplitude_scale" in src:
            n = self.number(src["amplitude_scale"], "audio.amplitude_scale", 0.0, 2.0)
            if n is not None:
                out["amplitude_scale"] = n

    def playback(self, src: Dict[str, Any], out: Dict[str, Any]):
        if "mode" in src:
            if src["mode"] not in PLAYBACK_MODES:
                self.err("playback.mode", f"must be one of {', '.join(PLAYBACK_MODES)}")
            else:
                out["mode"] = src["mode"]
        if "auto_advance" in src:
            b = self.boolean(src["auto_advance"], "playback.auto_advance")
            if b is not None:
                out["auto_advance"] = b

    def axis(self, src: Any, path: str) -> Optional[Dict[str, Any]]:
        if not isinstance(src, dict):
            self.err(path, "must be an object")
            return None
        waveform = src.get("waveform", "sine")
        if waveform not in WAVEFORMS:
            self.err(f"{path}.waveform", f"must be one of {', '.join(WAVEFORMS)}")

        if "keyframes" in src:
            raw = src["keyframes"]
            if not isinstance(raw, list) or len(raw) == 0:
                self.err(f"{path}.keyframes", "must be a non-empty list")
                return None
        else:
            # Legacy form: frequency/phase/amplitude directly on the axis,
            # possibly as single-element lists.
            def first(v, default):
                if isinstance(v, list):
                    return v[0] if v else default
                return v
            raw = [{
                "frequency": first(src.get("frequency", 1.0), 1.0),
                "phase": first(src.get("phase", 0.0), 0.0),
                "amplitude": first(src.get("amplitude", 0.9), 0.9),
            }]

        multi = len(raw) > 1
        keyframes = []
        for i, kf in enumerate(raw):
            kpath = f"{path}.keyframes[{i}]"
            if not isinstance(kf, dict):
                self.err(kpath, "must be an object")
                continue
            if multi and "duration" not in kf:
                self.err(f"{kpath}.duration", "is required when an axis has more than one keyframe")
            out_kf = {}
            for field, default in (("frequency", 1.0), ("phase", 0.0),
                                   ("amplitude", 0.8), ("duration", 1.0)):
                v = self.kf_value(kf.get(field, default), f"{kpath}.{field}", field)
                if v is not None:
                    out_kf[field] = v
            keyframes.append(out_kf)
        return {"waveform": waveform, "keyframes": keyframes}

    def signal_generator(self, src: Dict[str, Any], out: Dict[str, Any]):
        if "enabled" in src:
            b = self.boolean(src["enabled"], "signal_generator.enabled")
            if b is not None:
                out["enabled"] = b
        if "transition_time" in src:
            n = self.number(src["transition_time"], "signal_generator.transition_time", 0.0, 60.0)
            if n is not None:
                out["transition_time"] = n
        for axis in ("x", "y"):
            if axis in src:
                a = self.axis(src[axis], f"signal_generator.{axis}")
                if a is not None:
                    out[axis] = a

    def web(self, src: Dict[str, Any], out: Dict[str, Any]):
        if "port" in src:
            n = self.number(src["port"], "web.port", 1, 65535, integer=True)
            if n is not None:
                out["port"] = n


def validate_and_normalize(doc: Any) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """Validate a config document and return (normalized_doc, errors).

    The normalized document has every section and field filled in from
    DEFAULTS. On any error the document is None and errors lists every
    problem found as {"path": dotted.path, "message": text}.
    """
    v = _Validator()
    if not isinstance(doc, dict):
        return None, [{"path": "", "message": "config must be a JSON object"}]

    out = copy.deepcopy(DEFAULTS)

    if "input_directory" in doc:
        if not isinstance(doc["input_directory"], str) or not doc["input_directory"]:
            v.err("input_directory", "must be a non-empty string")
        else:
            out["input_directory"] = doc["input_directory"]

    v.display(v.section(doc, "display"), out["display"])
    v.visualization(v.section(doc, "visualization"), out["visualization"])
    v.audio(v.section(doc, "audio"), out["audio"])
    v.playback(v.section(doc, "playback"), out["playback"])
    v.signal_generator(v.section(doc, "signal_generator"), out["signal_generator"])
    v.web(v.section(doc, "web"), out["web"])

    if v.errors:
        return None, v.errors
    return out, []


def restart_required_changes(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """Return the restart-required field paths whose values differ."""
    changed = []
    for path in RESTART_REQUIRED_FIELDS:
        section, key = path.split(".")
        if old.get(section, {}).get(key) != new.get(section, {}).get(key):
            changed.append(path)
    return changed


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, doc: Dict[str, Any], _before_replace: Optional[Callable] = None):
    """Write JSON so the target is always either the old or the new complete file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(doc, indent=2) + "\n"
    with open(tmp, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if _before_replace is not None:
        _before_replace()   # test hook to simulate a crash before the swap
    os.replace(tmp, path)


class DebouncedWriter:
    """Writes the latest document at most once per `interval` seconds.

    update() records the newest document; tick() (called regularly from the
    main loop) writes it when due; flush() writes immediately.
    """

    def __init__(self, path: Path, interval: float = 1.0,
                 clock: Callable[[], float] = time.monotonic,
                 write: Callable[[Path, Dict[str, Any]], None] = atomic_write_json):
        self.path = Path(path)
        self.interval = interval
        self.clock = clock
        self._write = write
        self._pending: Optional[Dict[str, Any]] = None
        self._last_write = -math.inf
        self._lock = threading.Lock()

    def update(self, doc: Dict[str, Any]):
        with self._lock:
            self._pending = copy.deepcopy(doc)
        self.tick()

    def tick(self):
        with self._lock:
            if self._pending is None:
                return
            if self.clock() - self._last_write < self.interval:
                return
            doc, self._pending = self._pending, None
            self._last_write = self.clock()
        self._safe_write(doc)

    def flush(self):
        with self._lock:
            doc, self._pending = self._pending, None
            if doc is not None:
                self._last_write = self.clock()
        if doc is not None:
            self._safe_write(doc)

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def _safe_write(self, doc):
        try:
            self._write(self.path, doc)
        except OSError as e:
            print(f"ERROR: could not write {self.path}: {e}")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class PresetError(Exception):
    def __init__(self, message: str, status: int = 400):
        self.status = status
        super().__init__(message)


class PresetStore:
    """Every config/*.json except active.json is a named preset."""

    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)

    def _path(self, name: str) -> Path:
        if not isinstance(name, str) or not PRESET_NAME_RE.match(name):
            raise PresetError(f"invalid preset name {name!r}; use letters, digits, _ and -")
        if name == ACTIVE_NAME:
            raise PresetError(f"'{ACTIVE_NAME}' is reserved")
        return self.config_dir / f"{name}.json"

    def list(self) -> List[str]:
        names = [p.stem for p in self.config_dir.glob("*.json")
                 if p.stem != ACTIVE_NAME and PRESET_NAME_RE.match(p.stem)]
        return sorted(names)

    def load(self, name: str) -> Dict[str, Any]:
        path = self._path(name)
        if not path.exists():
            raise PresetError(f"preset {name!r} not found", 404)
        try:
            with open(path) as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise PresetError(f"preset {name!r} is not valid JSON: {e}")
        doc, errors = validate_and_normalize(raw)
        if errors:
            raise ValidationError(errors)
        return doc

    def save(self, name: str, doc: Dict[str, Any], overwrite: bool = False):
        path = self._path(name)
        if path.exists() and not overwrite:
            raise PresetError(f"preset {name!r} already exists", 409)
        atomic_write_json(path, doc)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None, [{"path": "", "message": f"file not found: {path}"}]
    except json.JSONDecodeError as e:
        return None, [{"path": "", "message": f"invalid JSON in {path}: {e}"}]
    return validate_and_normalize(raw)


def _log_errors(source: Path, errors: List[Dict[str, str]]):
    print(f"Config errors in {source}:")
    for e in errors:
        print(f"  {e['path'] or '(document)'}: {e['message']}")


def load_startup_config(config_dir: Path, cli_path: Optional[str] = None,
                        reset: bool = False) -> Dict[str, Any]:
    """Resolve the config to run with and make sure active.json reflects it.

    Precedence: a valid active.json wins unless reset is set. Otherwise the
    CLI path seeds active.json (a directory argument means a music folder),
    falling back to defaults.
    """
    config_dir = Path(config_dir)
    active_path = config_dir / f"{ACTIVE_NAME}.json"

    if not reset and active_path.exists():
        doc, errors = _read_json(active_path)
        if doc is not None:
            print(f"Loaded active configuration from {active_path}")
            return doc
        _log_errors(active_path, errors)
        print("Falling back to command-line config or defaults")

    doc = None
    if cli_path:
        p = Path(cli_path)
        if p.is_dir():
            doc = copy.deepcopy(DEFAULTS)
            doc["input_directory"] = str(p)
        else:
            doc, errors = _read_json(p)
            if doc is None:
                _log_errors(p, errors)
            else:
                print(f"Seeded active configuration from {p}")
    if doc is None:
        doc = copy.deepcopy(DEFAULTS)
        print("Using default configuration")

    config_dir.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_json(active_path, doc)
    except OSError as e:
        print(f"ERROR: could not write {active_path}: {e}")
    return doc
