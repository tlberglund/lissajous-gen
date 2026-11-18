import copy

import numpy as np
import pytest

from signal_generator import SignalGenerator

SR = 44100


def axis(freq=3.0, phase=0.0, amp=0.8, waveform="sine", keyframes=None):
    if keyframes is None:
        keyframes = [{"frequency": freq, "phase": phase, "amplitude": amp, "duration": 1.0}]
    return {"waveform": waveform, "keyframes": keyframes}


def gen_config(x=None, y=None, transition_time=1.5):
    return {"enabled": True, "transition_time": transition_time,
            "x": x or axis(3.0), "y": y or axis(4.0)}


def run(gen, seconds, fps):
    """Drive the generator like the render loop; return concatenated samples."""
    out = []
    for _ in range(int(round(seconds * fps))):
        out.append(gen.get_buffer(1.0 / fps))
    return np.concatenate(out) if out else np.zeros((0, 2))


def x_freq(gen):
    return gen.instantaneous_params()[0][0]


# -- 3.1 wall-clock timing --------------------------------------------------

@pytest.mark.parametrize("fps", [30, 60])
def test_signal_time_tracks_wall_clock(fps):
    # 30 wall-clock seconds of frames -> 30 seconds of samples
    gen = SignalGenerator(gen_config(), SR, 2048)
    total = sum(gen.samples_for(1.0 / fps) for _ in range(30 * fps))
    assert total == pytest.approx(30 * SR, rel=0.02)
    # And get_buffer's own bookkeeping agrees
    gen2 = SignalGenerator(gen_config(), SR, 2048)
    run(gen2, 1.0, fps)
    assert gen2.current_time == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("fps", [30, 60])
def test_keyframe_cycle_is_wall_clock_seconds(fps):
    """A 3-keyframe loop summing to 0.6 s takes 0.6 s of wall time to cycle."""
    kfs = [{"frequency": f, "phase": 0.0, "amplitude": 0.8, "duration": 0.2}
           for f in (2.0, 5.0, 8.0)]
    gen = SignalGenerator(gen_config(x=axis(keyframes=kfs)), SR, 2048)
    run(gen, 0.6, fps)
    assert gen.x.time == pytest.approx(0.6, rel=0.02)
    # Back at the start of the loop: frequency is the first keyframe's again
    assert x_freq(gen) == pytest.approx(2.0, abs=0.2)


def test_long_frame_is_capped():
    gen = SignalGenerator(gen_config(), SR, 2048)
    buf = gen.get_buffer(5.0)
    assert buf.shape == (2048, 2)
    # The backlog is dropped, not carried into the next frame
    assert gen.get_buffer(1.0 / 30).shape[0] in (1470, 1471)


def test_fractional_carry():
    gen = SignalGenerator(gen_config(), SR, 2048)
    counts = [gen.samples_for(1.0 / 60) for _ in range(60)]
    assert sum(counts) == SR
    assert set(counts) <= {735}


def test_no_dt_gives_full_buffer():
    gen = SignalGenerator(gen_config(), SR, 1024)
    assert gen.get_buffer().shape == (1024, 2)


# -- 3.2 glide --------------------------------------------------------------

def test_frequency_glide():
    gen = SignalGenerator(gen_config(transition_time=2.0), SR, 4096)
    run(gen, 0.1, 30)
    gen.retarget(gen_config(x=axis(5.0), transition_time=2.0))

    freqs = []
    for _ in range(int(2.5 * 30)):
        gen.get_buffer(1.0 / 30)
        freqs.append(x_freq(gen))
    glide = freqs[:59]
    assert all(b >= a for a, b in zip(glide, glide[1:]))
    assert glide[0] > 3.0 and glide[-1] < 5.0
    assert freqs[30] == pytest.approx(4.0, abs=0.1)        # halfway at 1 s
    assert all(f == pytest.approx(5.0) for f in freqs[61:])  # holds after 2 s


def test_unchanged_axis_is_not_retargeted():
    gen = SignalGenerator(gen_config(), SR, 2048)
    run(gen, 0.5, 30)
    y_time = gen.y.time
    gen.retarget(gen_config(x=axis(5.0)))
    assert gen.y.glide is None
    assert gen.y.time == y_time


# -- 3.3 snap cases ---------------------------------------------------------

def test_zero_transition_snaps():
    gen = SignalGenerator(gen_config(transition_time=0.0), SR, 2048)
    run(gen, 0.2, 30)
    gen.retarget(gen_config(x=axis(5.0), transition_time=0.0))
    gen.get_buffer(1.0 / 30)
    assert x_freq(gen) == pytest.approx(5.0)


def test_waveform_change_snaps_only_that_axis():
    gen = SignalGenerator(gen_config(transition_time=2.0), SR, 2048)
    run(gen, 0.2, 30)
    gen.retarget(gen_config(x=axis(5.0), y=axis(4.0, waveform="square"),
                            transition_time=2.0))
    gen.get_buffer(1.0 / 30)
    assert gen.y.generator.__class__.__name__ == "SquareGenerator"
    assert gen.y.glide is None
    assert gen.x.glide is not None
    assert 3.0 < x_freq(gen) < 5.0


def test_transition_time_only_change_applies_without_glide():
    gen = SignalGenerator(gen_config(transition_time=1.5), SR, 2048)
    gen.retarget(gen_config(transition_time=0.5))
    assert gen.transition_time == 0.5
    assert gen.x.glide is None and gen.y.glide is None


# -- 3.4 chained retarget and glide into a loop -----------------------------

def test_retarget_during_glide_is_continuous():
    gen = SignalGenerator(gen_config(transition_time=2.0), SR, 2048)
    gen.retarget(gen_config(x=axis(7.0), transition_time=2.0))
    run(gen, 1.0, 30)
    mid = x_freq(gen)
    assert 3.0 < mid < 7.0
    gen.retarget(gen_config(x=axis(1.0), transition_time=2.0))
    assert x_freq(gen) == pytest.approx(mid)
    gen.get_buffer(1.0 / 30)
    assert abs(x_freq(gen) - mid) < 0.2


def test_glide_into_keyframe_loop_starts_at_first_keyframe():
    kfs = [{"frequency": f, "phase": 0.0, "amplitude": 0.8, "duration": 1.0}
           for f in (6.0, 9.0, 2.0)]
    gen = SignalGenerator(gen_config(transition_time=0.5), SR, 2048)
    gen.retarget(gen_config(x=axis(keyframes=kfs), transition_time=0.5))
    run(gen, 0.5, 60)
    assert gen.x.glide is None
    assert gen.x.time < 0.05
    assert x_freq(gen) == pytest.approx(6.0, abs=0.2)
    run(gen, 1.0, 60)
    assert x_freq(gen) == pytest.approx(9.0, abs=0.2)


# -- 3.5 beam continuity ----------------------------------------------------

def max_step(samples):
    return float(np.max(np.abs(np.diff(samples[:, 0]))))


def test_beam_continuity_across_frequency_retarget():
    hi = SignalGenerator(gen_config(x=axis(50.0)), SR, 4096)
    steady = max_step(run(hi, 0.5, 30))

    gen = SignalGenerator(gen_config(x=axis(20.0), transition_time=0.5), SR, 4096)
    before = run(gen, 0.3, 30)
    gen.retarget(gen_config(x=axis(50.0), transition_time=0.5))
    after = run(gen, 1.0, 30)
    across = np.concatenate([before, after])
    assert max_step(across) <= steady * 1.001


def test_naive_rebuild_would_jump():
    """Sanity check that the continuity test can fail: rebuilding resets phase."""
    gen = SignalGenerator(gen_config(x=axis(20.0)), SR, 4096)
    before = run(gen, 10 / 30, 30)   # 6.67 cycles: stops mid-swing
    fresh = SignalGenerator(gen_config(x=axis(50.0)), SR, 4096)
    after = run(fresh, 0.1, 30)
    hi = SignalGenerator(gen_config(x=axis(50.0)), SR, 4096)
    steady = max_step(run(hi, 0.5, 30))
    assert max_step(np.concatenate([before, after])) > steady
