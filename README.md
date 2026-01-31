# Lissajous Figure Generator

Do you want to be a mad scientist, or just look like one? Then this project is for you.

A lissajous figure is a pattern created on an oscilloscope, and it's an absolutely necessary piece of production design in the background of every mad scientist's lab in any film meant to be taken seriously. Your home office may not be such a film, but that doesn't mean you should slack off on your background.

Scopes, whether the old analog kind or the modern digital variety, typically have an option to plot two inputs as X and Y coordinates. When you pipe the same sinusoid into both inputs, you get a line on the screen. If you shift the phase of one input by 90˚, you get a circle on the screen. When you vary the frequency and phase of the two inputs by small multiples, you get the cool pattern you see here.

![Lissajous Figure](https://github.com/user-attachments/assets/499ca943-5153-4cb8-b90f-cf944f694d1a)

You can buy an analog scope on eBay and rig up some signal generators (he says as if this is a thing everyone does on Tuesday night), but running this long-term will burn the living daylights out of that poor old CRT. So why not do this digitally? And why not elaborate from sinusoids to carefully crafted audio files that do even cooler things than mere lissajous figures?

Why not indeed.

## What This Does

The project creates a graphical display that simulates an analog oscilloscope in X-Y mode, plotting audio inputs to make cool patterns. It includes the ability to play existing audio files or to generate its own sinusoids with arbitrary frequency, phase, and amplitude relationships. It also sends the signal to the audio output of whatever platform it's running on.

## How to Run This

I've tested on MacOS 26 (as of this writing) and on a Raspberry Pi 1.

### MacOS

I'll generously assume you've got Python installed. You've got this.

```sh
$ pip install -r requirements.txt
$ python main.py config.json
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
$ python main.py config.json
```

### MP3 Mode

The project doesn't yet include any MP3s, but if you'd like to pursue this path, [here are some really cool audio files](https://oscilloscopemusic.com/watch/n-spheres) you can buy for a reasonable price. These go well beyond mere out-of-phase sinusoids to an elaborate use of the L and R stereo channels to plot a number of seriously cool designs.


### Function Generator Mode

If you want actual lissajous figures, this is the mode for you. You can configure the sinusoids to be generated on the X and Y inputs of the virtual scope, including relative frequency, phase (of course, when is phase not relative?), and amplitude relationships between the two axes. You can even provide a list of sinusoid definitions as keyframes to be animated.

### Config File Docs

The project comes with a few example config files: one for music input, one for a simple use of the signal generator, and one more interesting example.

#### Music Mode
```json
{
  "input_directory": "./music",
  "display": {
    "resolution": "auto",
    "fullscreen": true
  },
  "visualization": {
    "trace_color": [0.2, 1.0, 0.3],
    "trace_width": 2.0,
    "persistence_time": 1.5,
    "decay_rate": 0.7,
    "glow_radius": 4,
    "glow_intensity": 1.5,
    "beam_sharpness": 3.0,
    "background_color": [0.0, 0.0, 0.0]
  },
  "audio": {
    "sample_rate": 44100,
    "buffer_size": 1024,
    "amplitude_scale": 0.9
  },
  "playback": {
    "mode": "sequential",
    "auto_advance": true
  }
}
```




#### Function Generator Mode (Simple)

`config-generator.json`
```json
{
  "display": {
    "resolution": "auto",
    "fullscreen": true
  },
  "visualization": {
    "trace_color": [0.2, 1.0, 0.3],
    "trace_width": 10.0,
    "persistence_time": 5.0,
    "decay_rate": 4.0,
    "glow_radius": 10.0,
    "glow_intensity": 1.0,
    "beam_sharpness": 4.0,
    "background_color": [0.0, 0.1, 0.0]
  },
  "signal_generator": {
    "enabled": true,
    "x": {
      "waveform": "sine",
       "frequency": 1.0, 
       "phase": "0.78", 
       "amplitude": 0.9
    },
    "y": {
      "waveform": "sine",
      "frequency": 3.0, 
      "phase": 0.0, 
      "amplitude": 0.9
    }
  }
}
```


#### Function Generator Mode (Keyframed)

`config-generator-keyframed.json`
```json
{
  "display": {
    "resolution": "auto",
    "fullscreen": true
  },
  "visualization": {
    "trace_color": [0.2, 1.0, 0.3],
    "trace_width": 10.0,
    "persistence_time": 5.0,
    "decay_rate": 4.0,
    "glow_radius": 10.0,
    "glow_intensity": 1.0,
    "beam_sharpness": 4.0,
    "background_color": [0.0, 0.1, 0.0]
  },
  "signal_generator": {
    "enabled": true,
    "x": {
      "waveform": "sine",
      "keyframes": [
        {"frequency": 3.0, "phase": 0.0, "amplitude": 0.9, "duration": 10.0},
        {"frequency": 5.0, "phase": 0.0, "amplitude": 0.9, "duration": 10.0},
        {"frequency": 4.0, "phase": 0.0, "amplitude": 0.7, "duration": 10.0}
      ]
    },
    "y": {
      "waveform": "sine",
      "keyframes": [
        {"frequency": 4.0, "phase": 0.0, "amplitude": 0.7, "duration": 7.0},
        {"frequency": 1.0, "phase": 1.57, "amplitude": 0.9, "duration": 7.0},
        {"frequency": 5.0, "phase": 0.0, "amplitude": 0.9, "duration": 7.0}
      ]
    }
  }
}
```


#### Function Generator Mode (rand() example)

`config-generator-random.json`
```json
{
  "display": {
    "resolution": "auto",
    "fullscreen": true
  },
  "visualization": {
    "trace_color": [0.2, 1.0, 0.3],
    "trace_width": 10.0,
    "persistence_time": 5.0,
    "decay_rate": 4.0,
    "glow_radius": 10.0,
    "glow_intensity": 1.0,
    "beam_sharpness": 4.0,
    "background_color": [0.0, 0.1, 0.0]
  },
  "signal_generator": {
    "enabled": true,
    "x": {
      "waveform": "sine",
      "keyframes": [
        {"frequency": 3.0, "phase": "rand(0,1.57)", "amplitude": 0.9, "duration": 5.0}
      ]
    },
    "y": {
      "waveform": "sine",
      "keyframes": [
        {"frequency": 1.0, "phase": 0.0, "amplitude": 0.9, "duration": 5.0}
      ]
    }
  }
}
```

#### Config Reference





