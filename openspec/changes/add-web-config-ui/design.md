## Context

See proposal.md for motivation. The current shape of the code that constrains this design:

- `Application` (`src/lissajous.py`) creates a pygame/moderngl context and then commits to **one** mode for the life of the process. `run()` branches into the generator loop, the music loop, or the demo loop. The mixer is initialized only for music mode.
- All OpenGL calls must happen on the main thread. `OscilloscopeRenderer` already exposes live setters (`set_decay_rate`, `set_glow_intensity`, `set_glow_radius`, `set_beam_sharpness`, `set_trace_color`). Other visual params (trace width, background color) are set only in the constructor.
- `SignalGenerator.get_buffer()` always produces `buffer_size` samples and advances `current_time` by `buffer_size / sample_rate`, independent of the real frame interval. This is the source of the keyframe-duration bug.
- `SignalGenerator` uses phase accumulators (`x_phase`, `y_phase`) plus a per-keyframe phase offset, so frequency changes are already continuous. Constructing a new generator resets those accumulators.
- Config is a plain dict/attribute bag loaded once by `Config.load()`. Errors are printed and swallowed, falling back to defaults.
- On the Pi the app is started by systemd at boot with a config path argument. Usually no keyboard is attached.

## Goals / Non-Goals

**Goals:**
- A config change from the browser reaches the wall within ~one frame of the server accepting it.
- No beam discontinuity on any generator parameter change except waveform type.
- Invalid input can never crash or blank the display.
- No new Python dependencies; no front-end build step; works on a LAN without internet.

**Non-Goals:**
- A browser-side preview or mirror of the trace.
- Authentication, TLS, or multi-user conflict resolution beyond last-write-wins.
- Uploading music files through the UI.
- Removing keyboard controls (they stay; the UI does not try to stay in sync with them).
- Optimizing the per-sample Python loop in the generator.

## Decisions

### 1. Server thread + command queue; the main loop owns all state mutation

```
 browser --HTTP--> [server thread]                      [main thread / GL]
                    parse JSON                           each frame:
                    validate + normalize  --ok-->          drain queue
                    store.desired = cfg      queue.put     apply commands
                    reply 200 {restart_required}           render
                    (400 + field errors on failure)        flip
```

- The server uses stdlib `http.server.ThreadingHTTPServer` on a daemon thread. It never touches the renderer, the generator, or the mixer.
- The handoff is a `queue.Queue` of small command objects: `ApplyConfig(cfg)`, `SetSource(...)`, `Transport(action, track?)`, `Restart()`. The main loop drains it non-blockingly at the top of each frame.
- If several `ApplyConfig` commands are queued (fast slider drags), the main loop applies only the last one. Commands are coalesced.
- **Alternative considered:** a lock around shared objects so the server mutates them directly. Rejected because GL-thread affinity makes that unsafe for the renderer, and a queue keeps every mutation on one thread.

### 2. Validation lives in one place, before the queue

A single `validate_and_normalize(cfg) -> (cfg, errors)` function is shared by startup load, `PUT /api/config`, and preset load/save. It:
- type-checks and range-checks every field (for example amplitude 0..1, frequency > 0, colors 0..1 per channel, durations > 0 when there are 2+ keyframes);
- converts numeric strings to numbers (`"0.78"` → `0.78`);
- leaves `rand(min, max)` as its canonical string, re-serialized as `rand(<min>, <max>)` with min ≤ max enforced;
- converts the legacy per-axis `frequency/phase/amplitude` form into a single-element `keyframes` array;
- returns errors as `[{path: "signal_generator.x.keyframes[2].frequency", message}]`, which the UI maps back to fields.

At startup, an invalid file now logs the specific errors and falls back to the last good `active.json` or defaults. It no longer silently drops the whole file on a JSON error.

### 3. Apply tiers, decided by diffing old vs. new config

The main loop diffs the incoming config against the applied one by section:

| Changed | Action |
|---|---|
| `visualization.*` | renderer setters. Add setters for `trace_width` and `background_color`, which are currently constructor-only. |
| `audio.amplitude_scale` | set on the audio processor / generator output |
| `signal_generator.x/y`, `transition_time` | `generator.retarget(cfg)` (see 4) |
| `signal_generator.enabled` (source) | switch source (see 6) |
| `playback.auto_advance` | set directly |
| `display.*`, `audio.sample_rate`, `audio.buffer_size`, `web.port` | persist only; response sets `restart_required: true` |

### 4. Glide = a synthetic leading keyframe, phase accumulators preserved

`SignalGenerator.retarget(new_cfg)`:
1. Sample each axis's current instantaneous `(freq, phase, amp)` at `current_time`.
2. Build new `AxisConfig`s from `new_cfg`.
3. If the transition time `T > 0` and the waveform is unchanged, prepend a one-shot **glide segment** to each axis: from the sampled params to the new first keyframe over `T` seconds. After the glide, the axis time base resets to 0 and loops the new keyframes normally. The glide segment is not part of the loop.
4. Keep `x_phase` and `y_phase` untouched.
5. If the waveform type changed for an axis, that axis skips the glide (snap). The other axis still glides.

`transition_time` is a new field under `signal_generator` (seconds, default 1.5, 0 = snap). A retarget during an in-progress glide samples the current mid-glide params, so chained edits stay smooth.

**Alternative considered:** crossfading two complete generators' outputs. Rejected because it blends two different figures (visually muddy) and doubles CPU on the Pi.

### 5. Signal time follows the wall clock

`get_buffer(dt)` takes the real frame interval and generates `n = dt * sample_rate` samples, carrying the fractional remainder across frames. `current_time` advances by exactly `n / sample_rate`. At 30 fps that is ~1470 samples per frame instead of 2048. The renderer already accepts variable-length buffers. `buffer_size` stays as the cap (`n` is clamped) so a long stall can't produce a huge buffer.

### 6. One main loop, swappable source

`run()` is restructured into a single loop over a `source` object with a common interface (`get_buffer(dt)`, `pause()`, `resume()`, `is_finished()`):

```
              +------------- main loop -------------+
              | drain queue -> apply                |
              | buf = source.get_buffer(dt)         |
              | renderer.render_frame(buf, dt)      |
              +------------------+------------------+
                                 |
               +-----------------+------------------+
               |                                    |
        GeneratorSource                       MusicSource
        (SignalGenerator)              (AudioProcessor + pygame.mixer,
                                        FileManager over input_directory)
```

- The mixer is initialized lazily on the first switch to music and then kept.
- Switching sources clears the persistence buffer, so the old figure doesn't ghost into the new one.
- The demo mode (no MP3s found) becomes "MusicSource with an empty playlist" and falls back to rendering silence. The UI shows the empty directory.
- The music directory is whatever `input_directory` is at startup. The UI lists it but cannot change it.

### 7. Config store layout and startup precedence

```
config/
  active.json                      <- the running config (gitignored), written by the app
  config-generator.json            <- presets (everything else in config/*.json)
  config-generator-keyframed.json
  config-generator-random.json
  config-music.json
```

- **Writes are atomic**: write `active.json.tmp`, `fsync`, then `os.replace`.
- **Writes are debounced** to at most one per second, flushed on exit and before re-exec. Live slider drags would otherwise rewrite the SD card many times a second.
- **Startup precedence**: if `config/active.json` exists and is valid, it wins. The positional config argument (as used in the README and the systemd unit) is used only to seed `active.json` when none exists. A new `--reset` flag forces reseeding from the argument. This means web edits survive reboots without editing the systemd unit.
- Preset names must match `[A-Za-z0-9_-]+` and map to `config/<name>.json`. `active` is reserved. Save-as refuses to overwrite unless the request says `overwrite: true`.
- Loading a preset goes through the same path as `PUT /api/config`, so it glides, and it may set `restart_required`.

### 8. Restart by self re-exec

`POST /api/restart` queues `Restart()`. The main loop then flushes the store, stops the mixer, runs `pygame.quit()`, and calls `os.execv(sys.executable, [sys.executable] + sys.argv)`. Because of the startup precedence rule, the re-exec'd process loads `active.json`. The HTTP port is released when the old process image is replaced; the server socket sets `SO_REUSEADDR` to avoid bind races.

**Alternative considered:** exit non-zero and let systemd restart. Rejected as the primary path because it doesn't work on the Mac and depends on the unit's `Restart=` policy. It remains the crash backstop.

### 9. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | control panel (static files from `src/web/`) |
| GET | `/api/config` | desired config + `restart_required` flag |
| PUT | `/api/config` | full config document; 200 or 400 with field errors |
| GET | `/api/presets` | list preset names |
| POST | `/api/presets/<name>/load` | load preset into active |
| POST | `/api/presets/<name>` | save current active as preset (`{overwrite}`) |
| GET | `/api/music` | track list, current track, playing/paused |
| POST | `/api/music/<action>` | `play` (`{track}`), `pause`, `resume`, `next`, `prev` |
| POST | `/api/restart` | re-exec |
| GET | `/api/status` | source, fps, uptime (for the panel's status readouts) |

PUT takes the whole document, not a patch. That keeps validation total and the semantics obvious. With multiple browsers, the last write wins, and the UI re-fetches config when its tab becomes visible.

Port: new `web.port` config field (default 8080). The server binds `0.0.0.0`. If the bind fails, the app logs the error and keeps running the display without the web UI. The display must never depend on the server.

### 10. Visual design: oscilloscope front panel, vector-display aesthetic

The page should look like the front panel of a scope whose graphics are drawn by the same beam.

- **Palette**: near-black panel (`#020802`), phosphor green strokes (`#33ff66`), dimmer green for inactive and graticule lines (`#0f4a1c`), amber (`#ffb000`) only for warnings and restart-required, red-orange for field errors. There is no dark/light switch. The panel is always dark, like the hardware.
- **Everything is stroke, nothing is filled**: controls, borders, and icons are inline SVG drawn with 1.5–2px strokes, rounded caps, and a soft phosphor bloom (`filter: drop-shadow(0 0 2px currentColor) drop-shadow(0 0 6px currentColor)`) at reduced opacity. Hover and active states raise brightness/bloom instead of changing fill.
- **Lettering**: all panel lettering (legends, section headers, numeric readouts, track and preset names) is set in a single-stroke vector font: Hershey Simplex, rendered as SVG paths. Only typed-entry fields use text, set in the system monospace stack, so there are no font files to host. All assets are served locally; no CDN.
- **Graticule**: each section sits in a frame with faint 8x10 graticule lines and tick marks along the edges, like a CRT bezel. Section headers read like panel legends ("CH1 · X", "CH2 · Y", "DISPLAY", "TIMEBASE · KEYFRAMES", "SOURCE").
- **Controls**:
  - Continuous values: vector rotary **knobs**. Drag vertically to turn; the tick arc lights up to the current value. Each knob has a numeric readout; tapping the readout opens a text field for exact entry.
  - Fixed/random toggle per keyframe field: a two-position slide switch labeled `FIX`/`RND`. In `RND`, the field shows two knobs or readouts (min, max).
  - Waveform: four vector glyph buttons (sine, triangle, saw, square) that draw the waveform itself.
  - Source: a rotary selector `GEN` / `MUSIC`.
  - Colors: three knobs (R, G, B), with the swatch drawn as a stroked circle in that color.
- **Keyframe editor**: per axis, a horizontal **timebase strip** shows the loop as segments whose widths are proportional to duration (random durations use their midpoint, drawn dashed). Tapping a segment selects that keyframe, and its controls open below. Add, duplicate, delete, and move left/right are icon buttons on the strip. This is a structural view, not a waveform preview.
- **Status**: top-right LED-style indicators: `LINK` (server reachable), `APPLIED` (blinks briefly on each accepted change), `ERR` (lights on a rejected field), `RESTART` (amber, with a button, when restart-required settings are pending).
- **Responsive**: a phone is the primary client. At narrow widths the sections stack vertically and knobs are at least 56px with 44px tap targets. On wide screens, CH1 and CH2 sit side by side like a two-channel scope.
- Static files: `src/web/index.html`, `panel.css`, `panel.js`, `vector.js` (stroke components), `hershey.js` (glyph paths). Plain ES modules, no framework, no build.

### 11. Live edit flow in the browser

```
 knob turn --> local state --> debounce 250ms --> PUT /api/config
                                                     |
                                    200 --> APPLIED blink, clear field errors
                                    400 --> mark fields by path, ERR lit,
                                            local state kept (not reverted)
                                    network fail --> LINK off, retry on next edit
```

Invalid local edits are kept so the user can finish typing. The wall keeps showing the last valid state.

## Risks / Trade-offs

- [Keyframe timing fix changes existing animations] → Called out as BREAKING in the proposal. Durations in shipped presets can be scaled by ~1/1.4 if the old Pi feel is preferred.
- [Per-sample Python loop cost on Pi 4] → The timing fix lowers samples per frame from 2048 to ~1470 at 30 fps, which helps. Vectorization is explicitly out of scope.
- [SD card wear from live edits] → Debounced writes (≤1/s) plus atomic replace.
- [Open LAN endpoint with no auth] → Accepted by the user. The worst case is someone changing the display. The server serves only files under `src/web/`, and preset names are strictly validated to prevent path traversal.
- [Server crash or bind failure takes down the display] → The server runs on a daemon thread with exceptions caught and logged per request. Bind failure is non-fatal.
- [Two clients fighting] → Last write wins; tabs re-fetch on visibility. Acceptable for a home setup.
- [Hershey font subset adds asset work] → Fallback is the monospace font for all text. The aesthetic degrades gracefully.
- [os.execv on Pi under systemd] → Same PID, so systemd keeps tracking it. Verify on hardware during implementation.

## Migration Plan

1. Deploy as usual (`git pull` on the Pi). On first start, `active.json` is seeded from the systemd unit's config argument.
2. No unit change is required. Recommend adding `Restart=always` if it's not already present.
3. Add `config/active.json` to `.gitignore`.
4. Rollback: check out the previous commit. Old code ignores `active.json` and the new `web`/`transition_time` fields.

## Open Questions

- Exact default `transition_time` (1.5 s proposed). Tune on the wall.
- Whether to rescale durations in the shipped keyframed preset after the timing fix. Decide by eye on hardware.
