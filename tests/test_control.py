import copy
from pathlib import Path

import numpy as np
import pytest

from config_store import DEFAULTS
from control import AppState, ApplyConfig, Restart, Transport, apply_config, drain
from signal_generator import SignalGenerator
from sources import EMPTY, GeneratorSource, MusicSource, Source


# -- fakes ------------------------------------------------------------------

class FakeRenderer:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda value: self.calls.append((name, value))
        raise AttributeError(name)


class FakeAudio:
    def __init__(self):
        self.amplitude_scale = 0.9
        self.loaded = None
        self.finished = False

    def load_file(self, path):
        self.loaded = path
        return True

    def seek(self, pos):
        pass

    def get_buffer(self):
        return np.ones((4, 2), dtype=np.float32)

    def is_finished(self):
        return self.finished


class FakeFiles:
    def __init__(self, names):
        self.playlist = [Path(n) for n in names]
        self.current_index = 0

    def scan_directory(self):
        return len(self.playlist)

    def get_current_file(self):
        if not self.playlist:
            return None
        return self.playlist[self.current_index]

    def next_track(self):
        if not self.playlist:
            return None
        self.current_index = (self.current_index + 1) % len(self.playlist)
        return self.playlist[self.current_index]

    def previous_track(self):
        if not self.playlist:
            return None
        self.current_index = (self.current_index - 1) % len(self.playlist)
        return self.playlist[self.current_index]


class FakeMixer:
    def __init__(self):
        self.log = []

    def load_and_play(self, path, start=0.0):
        self.log.append(("play", Path(path).name))

    def pause(self):
        self.log.append(("pause",))

    def unpause(self):
        self.log.append(("unpause",))

    def stop(self):
        self.log.append(("stop",))

    def get_pos_seconds(self):
        return 0.0


class FakeGenerator:
    def __init__(self):
        self.retargets = []

    def retarget(self, cfg):
        self.retargets.append(copy.deepcopy(cfg))


class FakeApp:
    def __init__(self):
        self.renderer = FakeRenderer()
        self.generator = FakeGenerator()
        self.music = MusicSource(FakeAudio(), FakeFiles(["a.mp3"]), FakeMixer())
        self.switches = []

    def switch_source(self, name):
        self.switches.append(name)


def doc(**changes):
    d = copy.deepcopy(DEFAULTS)
    for path, value in changes.items():
        section, key = path.split("__")
        d[section][key] = value
    return d


# -- 5.1 source interface ---------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: GeneratorSource(SignalGenerator(DEFAULTS["signal_generator"])),
    lambda: MusicSource(FakeAudio(), FakeFiles(["a.mp3"]), FakeMixer()),
])
def test_sources_satisfy_interface(make):
    src = make()
    assert isinstance(src, Source)
    assert src.name in ("generator", "music")
    src.start()
    buf = src.get_buffer(1 / 30)
    assert buf.ndim == 2 and buf.shape[1] == 2
    src.pause()
    assert src.get_buffer(1 / 30).shape == (0, 2)
    src.resume()
    assert src.get_buffer(1 / 30).shape[0] > 0
    assert src.is_finished() in (True, False)
    src.stop()


def test_music_source_transport():
    mixer = FakeMixer()
    m = MusicSource(FakeAudio(), FakeFiles(["a.mp3", "b.mp3", "c.mp3"]), mixer)
    assert m.tracks() == ["a.mp3", "b.mp3", "c.mp3"]
    assert m.current() is None
    assert m.play("b.mp3")
    assert m.current() == "b.mp3" and m.playing
    assert not m.play("zzz.mp3")
    assert m.current() == "b.mp3"
    m.next()
    assert m.current() == "c.mp3"
    m.prev()
    m.prev()
    assert m.current() == "a.mp3"
    assert ("play", "a.mp3") in mixer.log


def test_music_source_empty_directory_renders_nothing():
    m = MusicSource(FakeAudio(), FakeFiles([]), FakeMixer())
    m.start()
    assert m.tracks() == []
    assert m.get_buffer(1 / 30).shape == (0, 2)


def test_music_auto_advance_and_stop_at_end():
    audio = FakeAudio()
    m = MusicSource(audio, FakeFiles(["a.mp3", "b.mp3"]), FakeMixer(), auto_advance=True)
    m.start()
    audio.finished = True
    m.get_buffer(1 / 30)
    assert m.current() == "b.mp3"
    m.auto_advance = False
    m.get_buffer(1 / 30)
    assert m.paused


# -- 5.4 command queue ------------------------------------------------------

def test_drain_coalesces_apply_config():
    state = AppState(DEFAULTS)
    for n in range(10):
        state.commands.put(ApplyConfig({"n": n}))
    cmds = drain(state.commands)
    assert cmds == [ApplyConfig({"n": 9})]
    assert drain(state.commands) == []


def test_drain_keeps_other_commands_in_order():
    state = AppState(DEFAULTS)
    state.commands.put(ApplyConfig({"n": 1}))
    state.commands.put(Transport("next"))
    state.commands.put(ApplyConfig({"n": 2}))
    state.commands.put(Restart())
    assert drain(state.commands) == [Transport("next"), ApplyConfig({"n": 2}), Restart()]


# -- 5.5 apply dispatch -----------------------------------------------------

def test_visualization_changes_call_setters():
    app = FakeApp()
    new = doc(visualization__trace_color=[1.0, 0.5, 0.0], visualization__trace_width=6.0,
              visualization__background_color=[0.0, 0.1, 0.0])
    assert apply_config(app, DEFAULTS, new) == []
    assert sorted(app.renderer.calls) == [
        ("set_background_color", (0.0, 0.1, 0.0)),
        ("set_trace_color", (1.0, 0.5, 0.0)),
        ("set_trace_width", 6.0),
    ]
    assert app.generator.retargets == [] and app.switches == []


@pytest.mark.parametrize("key,setter", [
    ("decay_rate", "set_decay_rate"), ("glow_radius", "set_glow_radius"),
    ("glow_intensity", "set_glow_intensity"), ("beam_sharpness", "set_beam_sharpness")])
def test_each_numeric_visual_setter(key, setter):
    app = FakeApp()
    apply_config(app, DEFAULTS, doc(**{f"visualization__{key}": 2.5}))
    assert app.renderer.calls == [(setter, 2.5)]


def test_amplitude_scale_and_auto_advance():
    app = FakeApp()
    apply_config(app, DEFAULTS, doc(audio__amplitude_scale=0.5, playback__auto_advance=False))
    assert app.music.audio.amplitude_scale == 0.5
    assert app.music.auto_advance is False


def test_generator_change_retargets():
    app = FakeApp()
    new = copy.deepcopy(DEFAULTS)
    new["signal_generator"]["x"]["keyframes"][0]["frequency"] = 5.0
    apply_config(app, DEFAULTS, new)
    assert len(app.generator.retargets) == 1
    assert app.generator.retargets[0]["x"]["keyframes"][0]["frequency"] == 5.0


def test_transition_time_change_retargets():
    app = FakeApp()
    apply_config(app, DEFAULTS, doc(signal_generator__transition_time=0.25))
    assert app.generator.retargets[0]["transition_time"] == 0.25


def test_source_switch():
    app = FakeApp()
    apply_config(app, DEFAULTS, doc(signal_generator__enabled=True))
    apply_config(app, doc(signal_generator__enabled=True), DEFAULTS)
    assert app.switches == ["generator", "music"]


def test_restart_required_fields_are_reported_not_applied():
    app = FakeApp()
    new = doc(display__fullscreen=True, audio__buffer_size=1024, web__port=9000)
    changed = apply_config(app, DEFAULTS, new)
    assert sorted(changed) == ["audio.buffer_size", "display.fullscreen", "web.port"]
    assert app.renderer.calls == [] and app.generator.retargets == []


def test_restart_required_flag_tracks_boot_config():
    state = AppState(DEFAULTS)
    assert not state.restart_required()
    state.set_desired(doc(display__resolution=[800, 600]))
    assert state.restart_required()
    state.set_desired(DEFAULTS)          # changed back: nothing pending
    assert not state.restart_required()
