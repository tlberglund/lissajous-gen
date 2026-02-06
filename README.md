# Lissajous Figure Generator

Do you want to be a mad scientist, or just look like one? Then this project is for you.

A lissajous figure is a pattern created on an oscilloscope, and it's an absolutely necessary piece of production design in the background of every mad scientist's lab in any film meant to be taken seriously. Your home office may not be such a film, but I don't accept excuses—only _results_.

Scopes, whether the old analog kind or the modern digital variety, typically have an option to plot two inputs as X and Y coordinates. When you pipe the same sinusoid into both inputs, you get a line on the screen. If you shift the phase of one input by 90˚, you get a circle on the screen. When you vary the frequency and phase of the two inputs by small multiples, you get the cool pattern you see here.

![Lissajous Figure](https://github.com/user-attachments/assets/499ca943-5153-4cb8-b90f-cf944f694d1a)

You can buy an analog scope on eBay and rig up some signal generators (he says as if this is a thing everyone does on Tuesday night), but running this long-term will burn the living daylights out of that poor old CRT. So why not do this digitally? And why not elaborate from sinusoids to carefully crafted audio files that do even cooler things than mere lissajous figures?

Why not indeed.

## What This Does

The project creates a graphical display that simulates an analog oscilloscope in X-Y mode, plotting audio inputs to make cool patterns. It includes the ability to play existing audio files or to generate its own sinusoids with arbitrary frequency, phase, and amplitude relationships. It also sends the signal to the audio output of whatever platform it's running on.

## How to Run This

I've tested on MacOS 26 (as of this writing) and on a Raspberry Pi 4 and 5. It will run on a 1GB Pi, but it really needs a 5 to run properly.

### MacOS

I'll generously assume you've got Python installed. You've got this.

```sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
$ python src/lissajous.py config/config-generator.json
```

### Raspberry Pi

Again, I'll leave the Python in your hands. You are smart and capable, and people like you.

```sh
$ sudo apt install ffmpeg libsdl2-dev libsdl2-mixer-dev
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
$ pip install -U audioop-lts
$ python src/lissajous.py config/config-generator.json
```

### MP3 Mode

The project doesn't yet include any MP3s, but if you'd like to pursue this path, [here are some really cool audio files](https://oscilloscopemusic.com/watch/n-spheres) you can buy for a reasonable price. These go well beyond mere out-of-phase sinusoids to an elaborate use of the L and R stereo channels to plot a number of seriously cool designs. The generator will play the music while displaying it.


### Function Generator Mode

If you want actual lissajous figures, this is the mode for you. You can configure the sinusoids to be generated on the X and Y inputs of the virtual scope, including relative frequency, phase (of course, when is phase not relative?), and amplitude relationships between the two axes. You can even provide a list of sinusoid definitions as keyframes to be animated.

### Config File Docs

The generator is fairly configurable. There are several example config files included with the project, plus a full reference to all the parameters below.

* `config-music.json` generates patterns from a directory of MP3 files.
* `config-generator.json` uses the built-in function generator in the simplest way possible.
* `config-generator-keyframed.json` uses the animation feature of the function generator to transition between different sinusoids
* `config-generator-random.json` demonstrates the ability to randomize parameters in the function generator


#### Config Reference

All configuration options with their types, defaults, and descriptions.

---

##### Top-Level Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input_directory` | string | `null` | Path to directory containing audio files. Required for music mode; ignored when signal generator is enabled. |

---

##### `display`

Controls the application window.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `resolution` | string or [int, int] | `"auto"` | Window size. Use `"auto"` for automatic sizing, or specify `[width, height]` in pixels. |
| `fullscreen` | boolean | `false` | Run in fullscreen mode. |

---

##### `visualization`

Controls the oscilloscope trace appearance.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `trace_color` | [float, float, float] | `[0.2, 1.0, 0.3]` | RGB color of the trace. Values range from 0.0 to 1.0. |
| `trace_width` | float | `2.0` | Width of the trace line in pixels. |
| `persistence_time` | float | `1.5` | How long (in seconds) the trace remains visible before fading. |
| `decay_rate` | float | `0.7` | Speed at which the trace fades. Higher values = faster fade. |
| `glow_radius` | float | `8` | Size of the bloom/glow effect around the trace. |
| `glow_intensity` | float | `1.8` | Brightness of the glow effect. |
| `beam_sharpness` | float | `3.0` | Sharpness of the beam center. Higher values = sharper, more focused beam. |
| `background_color` | [float, float, float] | `[0.0, 0.0, 0.0]` | RGB background color. Values range from 0.0 to 1.0. |

---

##### `audio`

Controls audio processing parameters.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sample_rate` | int | `44100` | Audio sample rate in Hz. |
| `buffer_size` | int | `2048` | Audio buffer size in samples. Smaller values reduce latency but may cause audio glitches. |
| `amplitude_scale` | float | `0.9` | Scales the input amplitude. Values >1.0 amplify; <1.0 attenuate. |

---

##### `playback`

Controls audio file playback behavior (music mode only).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | string | `"sequential"` | Playback order. Currently only `"sequential"` is supported. |
| `auto_advance` | boolean | `true` | Automatically play the next track when current track ends. |

---

##### `signal_generator`

Enables built-in waveform generation instead of audio file input.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | boolean | `false` | Enable signal generator mode. When `true`, audio files are ignored. |
| `x` | object | *(see below)* | Configuration for the X-axis signal. |
| `y` | object | *(see below)* | Configuration for the Y-axis signal. |

---

##### `signal_generator.x` / `signal_generator.y`

Each axis can be configured independently.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `waveform` | string | `"sine"` | Waveform type: `"sine"`, `"triangle"`, `"sawtooth"`, or `"square"`. |
| `keyframes` | array | *(single keyframe)* | Array of keyframe objects defining the signal parameters over time. |

**Legacy format:** You can also specify `frequency`, `phase`, and `amplitude` directly on the axis object (without `keyframes`), but the keyframes format is preferred.

---

##### Keyframe Objects

Each keyframe defines signal parameters for a segment of time. The signal smoothly interpolates between consecutive keyframes.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `frequency` | float or string | `1.0` | Frequency in Hz. Accepts `rand(min, max)` for randomization. |
| `phase` | float or string | `0.0` | Phase offset in radians. Accepts `rand(min, max)` for randomization. |
| `amplitude` | float or string | `0.8` | Signal amplitude (0.0 to 1.0). Accepts `rand(min, max)` for randomization. |
| `duration` | float or string | `1.0` | Duration of this keyframe segment in seconds. Required when multiple keyframes are defined. Accepts `rand(min, max)` for randomization. |

---

##### Random Values

Any keyframe parameter can use the `rand(min, max)` syntax to generate a random value within a range. Random values are reevaluated when transitioning between keyframes.

```json
{
  "frequency": "rand(1.0, 5.0)",
  "phase": "rand(0, 3.14159)",
  "amplitude": 0.9,
  "duration": 5.0
}
```

---

##### Keyframe Behavior

- **Single keyframe:** The signal uses constant parameters. Random values are reevaluated at each cycle boundary.
- **Multiple keyframes:** Parameters smoothly interpolate from one keyframe to the next. Random values are reevaluated one keyframe ahead to ensure smooth transitions.
- **Looping:** Keyframes loop continuously. The last keyframe interpolates back to the first.

---

##### Waveform Types

| Waveform | Description |
|----------|-------------|
| `sine` | Smooth sinusoidal wave. Classic oscilloscope look. |
| `triangle` | Linear ramps up and down. Sharper than sine. |
| `sawtooth` | Linear ramp up, instant drop. Creates asymmetric patterns. |
| `square` | Alternates between +1 and -1. Creates rectangular patterns. |

