## 1. Test Setup

- [x] 1.1 Add `requirements-dev.txt` with `pytest`, create `tests/` with a `conftest.py` that puts `src/` on the import path, and verify `pytest` runs and collects zero tests without error
- [x] 1.2 Add `config/active.json` and `config/*.tmp` to `.gitignore` and verify `git check-ignore config/active.json` reports it as ignored

## 2. Config Store

- [x] 2.1 Create `src/config_store.py` with `validate_and_normalize(doc) -> (doc, errors)` covering types and ranges for every documented field, plus `web.port`, `signal_generator.transition_time`, and the 2+ keyframe duration rule; verify with pytest cases for each error scenario in `specs/config-store/spec.md` (out-of-range amplitude, malformed `rand(5,`, missing duration, unknown waveform, negative transition time), each asserting the exact dotted error path
- [x] 2.2 Add normalization (numeric strings to numbers, canonical `rand(min, max)` with swapped bounds, legacy axis form to `keyframes`) and verify with pytest, including that all four shipped `config/*.json` files normalize without errors and `config-generator.json`'s `"0.78"` phase becomes `0.78`
- [x] 2.3 Implement atomic write (temp file, fsync, `os.replace`) and verify with a pytest that simulates a failure between write and replace and asserts the original file is intact
- [x] 2.4 Implement the debounced writer (at most one write per second, explicit `flush()`) and verify with a pytest using an injectable clock that ten rapid updates produce one write containing the last value, and that `flush()` writes immediately
- [x] 2.5 Implement presets (list excluding `active`, load, save-as with an overwrite flag, name regex `[A-Za-z0-9_-]+`, `active` reserved) and verify with pytest against a temp config dir, including rejection of `../etc/passwd`, `active`, and an overwrite without the flag
- [x] 2.6 Implement startup precedence (valid `active.json` wins; otherwise seed from the CLI path or defaults; `--reset` forces reseeding; invalid `active.json` logs each error path and falls back) and verify with pytest covering all four scenarios in the spec
- [x] 2.7 Replace `Config.load()` in `src/lissajous.py` with the store, keeping the existing attribute interface used by `Application`, and verify `python src/lissajous.py config/config-generator.json` still starts and creates `config/active.json`

## 3. Signal Generator Timing and Glide

- [x] 3.1 Change `SignalGenerator.get_buffer()` to `get_buffer(dt)`, generating `dt * sample_rate` samples with fractional carry and clamped to `buffer_size`; verify with pytest that 30 simulated seconds at 30 fps and at 60 fps both advance `current_time` by 30 s (within 2%), and that a 5 s stall yields at most `buffer_size` samples
- [x] 3.2 Implement `retarget(new_cfg)`: sample current per-axis params, build new `AxisConfig`s, keep `x_phase`/`y_phase`, and prepend a one-shot glide segment when `transition_time > 0`; verify with pytest that a 3 Hz to 5 Hz change with T=2 s produces monotonically rising instantaneous frequency reaching 5 Hz at 2 s and then holding
- [x] 3.3 Handle snap cases (transition time 0, waveform type change on one axis only, transition-time-only change) and verify with pytest that the snapped axis switches on the next buffer while the other axis still glides
- [x] 3.4 Handle retarget during an in-progress glide and glide into a multi-keyframe loop; verify with pytest that there is no parameter discontinuity at the second retarget and that the loop starts at keyframe 0 after the glide
- [x] 3.5 Add a continuity test: across a frequency retarget with T>0, the max delta between consecutive samples is no larger than the steady-state max delta at the higher frequency; verify it passes

## 4. Renderer Live Setters

- [x] 4.1 Add `set_trace_width()` and `set_background_color()` to `OscilloscopeRenderer` (alongside the existing setters) and verify by running the app on the Mac and calling them from a temporary keyboard binding or the debugger; the trace width and background change without restart

## 5. Main Loop, Sources, and Live Apply

- [x] 5.1 Define a source interface (`get_buffer(dt)`, `pause()`, `resume()`, `is_finished()`, `name`) and wrap `SignalGenerator` as `GeneratorSource`; verify with pytest that `GeneratorSource` satisfies the interface
- [x] 5.2 Create `MusicSource` around `AudioProcessor`, `FileManager`, and a lazily initialized `pygame.mixer`, with `play(track)`, `next`, `prev`, `pause`, `resume`, `tracks()`, and `current()`; an empty directory renders silence instead of entering demo mode; verify by running with `config/config-music.json` and an MP3 in `music/` (audio plays and the trace renders), and with an empty `music/` (the app runs and shows no trace)
- [x] 5.3 Restructure `Application.run()` into a single loop over the current source, removing `_run_signal_generator_mode` and `_run_demo_mode`, and verify both shipped generator presets and the music preset still run correctly on the Mac
- [x] 5.4 Add the command queue (`ApplyConfig`, `SetSource`, `Transport`, `Restart`) drained non-blockingly each frame, with `ApplyConfig` coalesced to the last one; verify with pytest using a fake renderer and source that ten queued `ApplyConfig`s result in exactly one apply of the tenth
- [x] 5.5 Implement section diffing and dispatch per the design's apply-tier table (renderer setters, `amplitude_scale`, generator `retarget`, source switch with persistence clear, `auto_advance`, restart-required flagging), and reject `input_directory` changes; verify with pytest against fakes for each row of the table
- [x] 5.6 Implement restart (flush store, stop mixer, `pygame.quit()`, `os.execv` with the original argv) and verify on the Mac that triggering it brings the app back with `active.json` values, including a changed `fullscreen`

## 6. HTTP Server and API

- [x] 6.1 Create `src/web_server.py` with a `ThreadingHTTPServer` on a daemon thread bound to `0.0.0.0:web.port` with `SO_REUSEADDR`, per-request exception handling (500 + log), and non-fatal bind failure; verify with pytest that a second server on the same port logs an error without raising, and that a handler exception returns 500
- [x] 6.2 Serve static files from `src/web/` with path-traversal protection and verify with pytest that `/` returns `index.html` and `/../config/active.json` returns 404
- [x] 6.3 Implement `GET`/`PUT /api/config` (full document; 200 with `restart_required`; 400 with `{path, message}` errors for validation failures and malformed JSON; enqueue `ApplyConfig` on success) and verify with pytest using a real server on an ephemeral port and a fake queue
- [x] 6.4 Implement the preset endpoints (list, load, save with 409 on an overwrite conflict) and verify with pytest
- [x] 6.5 Implement `GET /api/music` and `POST /api/music/<action>` (404 for unknown tracks; play switches the source) and `GET /api/status`, reading app state through a thread-safe snapshot updated by the main loop; verify with pytest
- [x] 6.6 Implement `POST /api/restart` and verify end to end on the Mac with `curl -X POST localhost:8080/api/restart`: the app restarts and `curl localhost:8080/api/status` responds again
- [x] 6.7 Start the server from `Application` and verify end to end on the Mac: `curl -X PUT` with a changed `glow_intensity` visibly changes the display within a frame, and an invalid document returns 400 with no visible change

## 7. Control Panel: Vector Foundation

- [x] 7.1 Create `src/web/index.html`, `panel.css`, and `panel.js` (ES modules, no build) with the palette tokens, dark panel, bloom filter, and graticule section frames; verify the page loads from the running app with no external requests in the browser's network tab
- [x] 7.2 Add a Hershey Simplex glyph subset (`hershey.js`) and an SVG text helper used for all panel lettering, including legends, readouts, and track names (typed-entry fields use the system monospace stack, so no font files are needed); verify that headers and readouts render as stroked paths, with no external requests
- [ ] 7.3 Build the vector knob component (SVG arc ticks, vertical-drag turning, touch support, value readout, tap-to-type exact entry, min/max/step, degree display option); verify in a scratch page on desktop and on a phone that dragging and typing both work and report values
- [ ] 7.4 Build the FIX/RND slide switch, the waveform glyph buttons (sine, triangle, saw, square drawn as their waveforms), the GEN/MUSIC rotary selector, and the LED indicators (LINK, APPLIED, ERR, RESTART); verify each renders as stroke-only vector art and responds to touch

## 8. Control Panel: Sections and Behavior

- [x] 8.1 Implement the config client: fetch on load and on tab visibility, keep local state, debounce 250 ms, PUT the full document, and drive the LINK/APPLIED/ERR/RESTART indicators; verify against the running app that a knob turn reaches the wall and APPLIED blinks, and that stopping the app turns LINK off
- [x] 8.2 Build the DISPLAY section (all visualization settings, RGB knobs with stroked swatches for both colors) and the restart-required settings (resolution, fullscreen, sample rate, buffer size) marked "applies on restart"; verify each control changes the wall live except the marked ones, which light RESTART
- [x] 8.3 Build the CH1 · X and CH2 · Y sections: waveform selector plus a keyframe timebase strip with proportional segments, dashed segments for random durations, selection, and add/duplicate/delete/move controls (delete disabled on the last keyframe); verify against `config-generator-keyframed` that the strip ratios match the durations and that edits glide on the wall
- [x] 8.4 Build the selected-keyframe editor (frequency, phase in degrees, amplitude, duration, each with FIX/RND and min/max in RND), plus the transition-time knob; verify that switching to RND initializes min=max=current, and that 90 degrees is stored as about 1.5708 in `active.json`
- [x] 8.5 Implement field-level error display mapped from server error paths (highlight and message, keep the user's input, clear on the next success); verify by typing an out-of-range amplitude that the field highlights while the wall is unchanged
- [x] 8.6 Build the SOURCE section (GEN/MUSIC selector, track list from `/api/music`, current-track marker, play/pause/next/prev, auto-advance toggle); verify on the Mac with MP3s in `music/` that selecting a track switches the source and plays it
- [x] 8.7 Build the presets UI (list, load, save-as with an overwrite confirmation); verify that loading `config-generator-random` updates the controls and the wall glides to it, and that saving over an existing name prompts first
- [ ] 8.8 Make the layout responsive (stacked sections at 375 px with 44 px minimum tap targets; side-by-side CH1/CH2 on wide screens); verify in browser devtools at 375 px that nothing scrolls horizontally, and on a real phone

## 9. Integration and Docs

- [ ] 9.1 Deploy to the Pi under systemd and verify: the panel is reachable from a phone on the LAN; live edits glide on the wall; restart via the panel works without systemd marking the unit failed; edits survive a power cycle; and the display keeps running when the panel is closed
- [ ] 9.2 Check keyframe timing on the Pi with a stopwatch against `config-generator-keyframed` (cycle length matches the summed durations within 2%), then decide by eye whether to rescale the shipped preset durations and settle the default `transition_time` (design Open Questions)
- [x] 9.3 Update `README.md` with the web UI section (URL and port, `web.port`, presets, `active.json` and `--reset`, `transition_time`, the duration-semantics change, the `Restart=always` recommendation) and verify the config reference tables include the new fields
