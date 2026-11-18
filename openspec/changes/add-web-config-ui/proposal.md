## Why

The generator runs headless on a Raspberry Pi, launched by systemd, usually with no keyboard attached. Changing anything means SSHing in, editing JSON, and restarting the process. A browser-based control panel on the local network makes the display remotely tunable, and changes should land on the wall live so the wall itself is the preview.

## What Changes

- Add an embedded HTTP server (stdlib only, bound to `0.0.0.0`, no auth) that runs on a background thread inside the visualizer process and serves a single-page control panel plus a JSON API.
- Add a control panel UI styled as an oscilloscope front panel with vector-display graphics. It edits visualization settings, the source (generator or music), the per-axis waveform, and a full per-axis keyframe editor where each value is either fixed or `rand(min, max)`.
- Apply config changes live: edits are debounced in the browser, validated on the server, queued, and applied by the main (OpenGL) loop. Invalid input is rejected with a field-level error and never reaches the display.
- Add **glide**: when signal-generator parameters change, the generator ramps from its current instantaneous frequency/phase/amplitude to the new keyframes over a configurable transition time (0 = snap). Phase accumulators are preserved so the beam never jumps. Waveform-type changes snap.
- Make the source switchable at runtime between the signal generator and music playback. Music mode lists files from a fixed music directory and supports play-selected, pause/resume, next, and previous.
- Settings that cannot be hot-applied (resolution, fullscreen, sample rate, buffer size) are persisted and then applied by the process re-executing itself.
- Introduce a config store: the running config persists to `config/active.json` with atomic writes; existing `config/*.json` files become **presets** that can be loaded and saved from the UI.
- **BREAKING** (behavioral): keyframe `duration` now means wall-clock seconds. Signal time previously advanced by `buffer_size / sample_rate` per rendered frame, which ran faster than real time (~1.4x on Pi at 30 fps, ~2.8x on Mac at 60 fps). Existing keyframed configs will animate more slowly than before.
- Normalize numeric values stored as strings (for example `"phase": "0.78"`) to JSON numbers on save. The `rand(min, max)` string syntax is unchanged.

## Capabilities

### New Capabilities
- `config-store`: Persisting the active config atomically, preset files (list, load, save-as), value normalization, and validation rules shared by every writer.
- `config-api`: The embedded HTTP server and JSON endpoints for reading/writing config, presets, source control, music track listing/transport, and restart.
- `web-config-ui`: The browser control panel, including the keyframe editor, live debounced apply, error display, and the oscilloscope/vector visual language.
- `live-config-apply`: How validated changes cross from the server thread to the render loop, which settings apply live vs. by re-exec, and runtime source switching between generator and music.
- `signal-generator`: Wall-clock-accurate signal time, glide transitions on retarget, and phase continuity across config changes.

### Modified Capabilities
<!-- None: no specs exist yet under openspec/specs/. -->

## Impact

- **Code**: `src/lissajous.py` (Config loading, Application main loop restructured around a swappable source and a command queue, keyboard handling left as-is), `src/signal_generator.py` (wall-clock time, retarget/glide), `src/audio_processor.py` (music source wrapper, track listing). New modules for the server, store, and static UI assets.
- **Config files**: `config/*.json` are treated as presets; new `config/active.json` is created on first run from the preset given on the command line (or defaults). The command-line argument keeps working.
- **Dependencies**: none added on the Python side (stdlib `http.server`, `threading`, `queue`, `json`). The UI is static HTML/CSS/JS served from the repo with no build step and no CDN, so it works on an offline LAN.
- **Deployment**: the systemd unit on the Pi needs no change for normal operation. Self re-exec handles restart-required settings, and `Restart=always` is still recommended as a crash backstop. The web port (default 8080) must be reachable on the LAN.
- **Behavior**: keyframe animation speed changes, as noted above.
