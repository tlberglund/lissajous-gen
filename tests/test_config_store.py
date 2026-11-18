import copy
import json
from pathlib import Path

import pytest

from config_store import (
    DEFAULTS, DebouncedWriter, PresetError, PresetStore, ValidationError,
    atomic_write_json, load_startup_config, restart_required_changes,
    validate_and_normalize,
)

REPO_CONFIG = Path(__file__).resolve().parent.parent / "config"


def gen_doc(**axis_overrides):
    doc = copy.deepcopy(DEFAULTS)
    doc["signal_generator"]["x"].update(axis_overrides)
    return doc


def error_paths(doc):
    out, errors = validate_and_normalize(doc)
    assert out is None
    return [e["path"] for e in errors]


# -- 2.1 validation ---------------------------------------------------------

def test_defaults_are_valid():
    out, errors = validate_and_normalize({})
    assert errors == []
    assert out == DEFAULTS


def test_out_of_range_amplitude():
    doc = gen_doc(keyframes=[{"frequency": 1, "phase": 0, "amplitude": 1.5}])
    assert error_paths(doc) == ["signal_generator.x.keyframes[0].amplitude"]


def test_malformed_rand():
    doc = gen_doc(keyframes=[
        {"frequency": 1, "duration": 2},
        {"frequency": "rand(5,", "duration": 2},
    ])
    assert error_paths(doc) == ["signal_generator.x.keyframes[1].frequency"]


def test_missing_duration_with_multiple_keyframes():
    doc = gen_doc(keyframes=[
        {"frequency": 1, "duration": 2},
        {"frequency": 2, "duration": 2},
        {"frequency": 3},
    ])
    assert error_paths(doc) == ["signal_generator.x.keyframes[2].duration"]


def test_unknown_waveform():
    doc = gen_doc(waveform="noise")
    assert error_paths(doc) == ["signal_generator.x.waveform"]


def test_negative_transition_time():
    doc = copy.deepcopy(DEFAULTS)
    doc["signal_generator"]["transition_time"] = -1
    assert error_paths(doc) == ["signal_generator.transition_time"]


def test_multiple_errors_reported_together():
    doc = copy.deepcopy(DEFAULTS)
    doc["visualization"]["trace_color"] = [0, 2, 0]
    doc["web"]["port"] = 70000
    doc["display"]["fullscreen"] = "yes"
    assert sorted(error_paths(doc)) == [
        "display.fullscreen", "visualization.trace_color[1]", "web.port"]


def test_bool_is_not_a_number():
    doc = copy.deepcopy(DEFAULTS)
    doc["visualization"]["trace_width"] = True
    assert error_paths(doc) == ["visualization.trace_width"]


def test_non_object_document():
    out, errors = validate_and_normalize([1, 2])
    assert out is None and errors[0]["path"] == ""


# -- 2.2 normalization ------------------------------------------------------

def test_numeric_string_phase():
    doc = gen_doc(keyframes=[{"frequency": 1, "phase": "0.78", "amplitude": 0.9}])
    out, _ = validate_and_normalize(doc)
    assert out["signal_generator"]["x"]["keyframes"][0]["phase"] == 0.78


def test_reversed_rand_bounds():
    doc = gen_doc(keyframes=[{"frequency": "rand(5, 1)"}])
    out, _ = validate_and_normalize(doc)
    assert out["signal_generator"]["x"]["keyframes"][0]["frequency"] == "rand(1, 5)"


def test_rand_canonical_spacing():
    doc = gen_doc(keyframes=[{"frequency": 1, "phase": "rand(0,1.57)"}])
    out, _ = validate_and_normalize(doc)
    assert out["signal_generator"]["x"]["keyframes"][0]["phase"] == "rand(0, 1.57)"


def test_legacy_axis_form():
    doc = copy.deepcopy(DEFAULTS)
    doc["signal_generator"]["x"] = {"waveform": "sine", "frequency": 2.0,
                                    "phase": "0.5", "amplitude": 0.7}
    out, errors = validate_and_normalize(doc)
    assert errors == []
    assert out["signal_generator"]["x"] == {
        "waveform": "sine",
        "keyframes": [{"frequency": 2.0, "phase": 0.5, "amplitude": 0.7, "duration": 1.0}],
    }


def test_resolution_forms():
    for given, expected in (("auto", "auto"), ("1920x1080", [1920, 1080]),
                            ([800, 600], [800, 600])):
        doc = copy.deepcopy(DEFAULTS)
        doc["display"]["resolution"] = given
        out, errors = validate_and_normalize(doc)
        assert errors == [] and out["display"]["resolution"] == expected


@pytest.mark.parametrize("name", [
    "config-generator", "config-generator-keyframed",
    "config-generator-random", "config-music"])
def test_shipped_configs_normalize(name):
    with open(REPO_CONFIG / f"{name}.json") as f:
        raw = json.load(f)
    out, errors = validate_and_normalize(raw)
    assert errors == []
    if name == "config-generator":
        assert out["signal_generator"]["x"]["keyframes"][0]["phase"] == 0.78


def test_restart_required_changes():
    a = copy.deepcopy(DEFAULTS)
    b = copy.deepcopy(DEFAULTS)
    b["display"]["fullscreen"] = True
    b["visualization"]["glow_intensity"] = 3.0
    assert restart_required_changes(a, b) == ["display.fullscreen"]


# -- 2.3 atomic write -------------------------------------------------------

def test_atomic_write_survives_crash_before_replace(tmp_path):
    target = tmp_path / "active.json"
    atomic_write_json(target, {"v": 1})

    def crash():
        raise KeyboardInterrupt("simulated kill")

    with pytest.raises(KeyboardInterrupt):
        atomic_write_json(target, {"v": 2}, _before_replace=crash)
    assert json.loads(target.read_text()) == {"v": 1}

    atomic_write_json(target, {"v": 3})
    assert json.loads(target.read_text()) == {"v": 3}


# -- 2.4 debounced writer ---------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_debounce_coalesces_and_flushes(tmp_path):
    clock = FakeClock()
    writes = []
    w = DebouncedWriter(tmp_path / "a.json", 1.0, clock, lambda p, d: writes.append(d))

    w.update({"n": 0})          # first write goes out immediately
    for n in range(1, 11):
        clock.t += 0.05
        w.update({"n": n})
    assert writes == [{"n": 0}]
    assert w.has_pending

    clock.t += 1.0
    w.tick()
    assert writes == [{"n": 0}, {"n": 10}]

    w.update({"n": 11})
    assert writes[-1] == {"n": 10}
    w.flush()
    assert writes[-1] == {"n": 11}
    assert not w.has_pending


# -- 2.5 presets ------------------------------------------------------------

def make_preset_dir(tmp_path):
    for name in ("config-generator", "config-generator-keyframed",
                 "config-generator-random", "config-music"):
        (tmp_path / f"{name}.json").write_text((REPO_CONFIG / f"{name}.json").read_text())
    (tmp_path / "active.json").write_text(json.dumps(DEFAULTS))
    return PresetStore(tmp_path)


def test_list_presets(tmp_path):
    store = make_preset_dir(tmp_path)
    assert store.list() == ["config-generator", "config-generator-keyframed",
                            "config-generator-random", "config-music"]


def test_load_preset(tmp_path):
    store = make_preset_dir(tmp_path)
    doc = store.load("config-generator-keyframed")
    assert len(doc["signal_generator"]["x"]["keyframes"]) == 3


def test_save_and_overwrite(tmp_path):
    store = make_preset_dir(tmp_path)
    store.save("spirals", DEFAULTS)
    assert (tmp_path / "spirals.json").exists()
    with pytest.raises(PresetError) as e:
        store.save("spirals", {"changed": True})
    assert e.value.status == 409
    assert json.loads((tmp_path / "spirals.json").read_text()) == DEFAULTS
    store.save("spirals", {"changed": True}, overwrite=True)
    assert json.loads((tmp_path / "spirals.json").read_text()) == {"changed": True}


@pytest.mark.parametrize("bad", ["../etc/passwd", "active", "", "a b", "x.json"])
def test_invalid_preset_names(tmp_path, bad):
    store = make_preset_dir(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(PresetError):
        store.save(bad, DEFAULTS)
    with pytest.raises(PresetError):
        store.load(bad)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_load_missing_preset(tmp_path):
    store = make_preset_dir(tmp_path)
    with pytest.raises(PresetError) as e:
        store.load("nope")
    assert e.value.status == 404


def test_load_invalid_preset(tmp_path):
    store = make_preset_dir(tmp_path)
    (tmp_path / "broken.json").write_text('{"web": {"port": -1}}')
    with pytest.raises(ValidationError):
        store.load("broken")


# -- 2.6 startup precedence -------------------------------------------------

def seed_file(tmp_path, freq):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({"signal_generator": {"x": {"keyframes": [{"frequency": freq}]}}}))
    return str(p)


def x_freq(doc):
    return doc["signal_generator"]["x"]["keyframes"][0]["frequency"]


def test_first_run_seeds_active(tmp_path):
    cfg = tmp_path / "config"
    doc = load_startup_config(cfg, seed_file(tmp_path, 7.0))
    assert x_freq(doc) == 7.0
    assert x_freq(json.loads((cfg / "active.json").read_text())) == 7.0


def test_active_wins_over_cli(tmp_path):
    cfg = tmp_path / "config"
    load_startup_config(cfg, seed_file(tmp_path, 7.0))
    doc = load_startup_config(cfg, seed_file(tmp_path, 9.0))
    assert x_freq(doc) == 7.0


def test_reset_reseeds(tmp_path):
    cfg = tmp_path / "config"
    load_startup_config(cfg, seed_file(tmp_path, 7.0))
    doc = load_startup_config(cfg, seed_file(tmp_path, 9.0), reset=True)
    assert x_freq(doc) == 9.0
    assert x_freq(json.loads((cfg / "active.json").read_text())) == 9.0


def test_invalid_active_falls_back(tmp_path, capsys):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "active.json").write_text(json.dumps({"web": {"port": "eighty"}}))
    doc = load_startup_config(cfg, seed_file(tmp_path, 5.0))
    assert x_freq(doc) == 5.0
    assert "web.port" in capsys.readouterr().out


def test_directory_argument_is_music_folder(tmp_path):
    music = tmp_path / "tunes"
    music.mkdir()
    doc = load_startup_config(tmp_path / "config", str(music))
    assert doc["input_directory"] == str(music)
