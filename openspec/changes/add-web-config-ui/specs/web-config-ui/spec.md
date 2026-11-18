## Purpose

Defines the browser control panel for remotely editing the visualizer on a local network, including the keyframe editor, live-as-you-edit behavior, and its oscilloscope front-panel visual language.

## ADDED Requirements

### Requirement: Editable settings
The control panel SHALL let the user edit every visualization setting, `audio.amplitude_scale`, `playback.auto_advance`, `signal_generator.transition_time`, the source (generator or music), each axis's waveform, and each axis's keyframes. It SHALL also let the user edit the restart-required display and audio settings, marked as requiring a restart.

#### Scenario: All visualization settings present
- **WHEN** the panel loads
- **THEN** it shows controls for trace color, trace width, decay rate, glow radius, glow intensity, beam sharpness, and background color, initialized from the current config

#### Scenario: Restart-required settings marked
- **WHEN** the panel shows resolution, fullscreen, sample rate, or buffer size
- **THEN** those controls are visibly marked as applying on restart

### Requirement: Keyframe editor
For each axis, the panel SHALL show the keyframe loop as a strip of segments whose widths are proportional to their durations. It SHALL let the user select a keyframe to edit; add, duplicate, and delete keyframes; and move a keyframe earlier or later. It SHALL NOT allow deleting an axis's last keyframe.

#### Scenario: Proportional strip
- **WHEN** an axis has keyframes of 10 s, 10 s, and 5 s
- **THEN** the strip shows three segments with widths in a 2:2:1 ratio

#### Scenario: Add keyframe
- **WHEN** the user adds a keyframe after the selected one
- **THEN** a new keyframe with the selected keyframe's values appears immediately after it and is selected

#### Scenario: Last keyframe protected
- **WHEN** an axis has one keyframe
- **THEN** the delete control for that keyframe is disabled

#### Scenario: Reorder
- **WHEN** the user moves the second keyframe earlier
- **THEN** it becomes the first keyframe and the change is applied to the wall

### Requirement: Fixed or random values
Each keyframe field (frequency, phase, amplitude, duration) SHALL have a fixed/random switch. In fixed mode it SHALL edit a single number. In random mode it SHALL edit a minimum and a maximum and store `rand(min, max)`.

#### Scenario: Switch to random
- **WHEN** the user switches a fixed frequency of 3.0 to random
- **THEN** the field shows min and max controls, both initialized to 3.0

#### Scenario: Random duration on the strip
- **WHEN** a keyframe's duration is random
- **THEN** its strip segment is drawn at the midpoint duration with a dashed outline

### Requirement: Phase entered in friendly units
Phase controls SHALL display and accept values in degrees, while the stored config remains in radians.

#### Scenario: Degrees to radians
- **WHEN** the user sets a phase of 90 degrees
- **THEN** the stored phase is approximately 1.5708

### Requirement: Live apply as you edit
The panel SHALL send the complete config to the server automatically after the user stops changing a control for 250 ms, with no separate apply step. It SHALL indicate briefly when a change is accepted.

#### Scenario: Knob turn reaches the wall
- **WHEN** the user turns the glow intensity knob and pauses
- **THEN** the change is sent within about 250 ms and the panel's applied indicator blinks

### Requirement: Field-level errors
When the server rejects a change, the panel SHALL highlight each field named in the errors and show its message. It SHALL keep the user's in-progress input rather than reverting it, and SHALL clear the highlights once a later change is accepted.

#### Scenario: Partial rand entry
- **WHEN** the user's input makes a field invalid and the server rejects the document
- **THEN** that field is highlighted with the error message, the input is unchanged, and the wall keeps showing the last valid config

### Requirement: Connection and restart indicators
The panel SHALL show whether the server is reachable, and SHALL show a restart-pending indicator with a restart control whenever the server reports `restart_required`.

#### Scenario: Server unreachable
- **WHEN** a request to the server fails
- **THEN** the link indicator turns off until a later request succeeds

#### Scenario: Restart pending
- **WHEN** the server reports `restart_required: true`
- **THEN** the panel shows the restart indicator and a control that requests a restart

### Requirement: Presets in the panel
The panel SHALL list presets, load a selected preset, and save the current config under a new or existing name, asking for confirmation before overwriting.

#### Scenario: Load preset
- **WHEN** the user loads `config-generator-random`
- **THEN** the panel's controls update to that preset's values and the wall glides to it

#### Scenario: Overwrite confirmation
- **WHEN** the user saves under an existing preset name
- **THEN** the panel asks for confirmation before overwriting

### Requirement: Music controls
When music is the source, or when the user chooses it, the panel SHALL list the tracks in the music directory, play a selected track, and offer pause/resume, next, and previous controls, showing which track is current.

#### Scenario: Play from list
- **WHEN** the user selects a track in the list
- **THEN** that track plays on the Pi and is marked as current in the panel

### Requirement: Stays current across tabs
The panel SHALL re-fetch the config and music state when its browser tab becomes visible again.

#### Scenario: Returning to the tab
- **WHEN** another client changed the config while this tab was in the background
- **THEN** this tab shows the new values once it becomes visible

### Requirement: Oscilloscope front-panel presentation
The panel SHALL look like an oscilloscope front panel drawn in vector-display style: a dark panel; all controls, frames, icons, and lettering drawn as glowing phosphor-green strokes without solid fills; rotary knobs with numeric readouts for continuous values; waveform selectors drawn as their waveforms; and section frames with graticule markings. Amber SHALL be reserved for warnings such as a pending restart.

#### Scenario: Vector rendering
- **WHEN** the panel is viewed
- **THEN** controls and labels are rendered as stroked vector graphics with a glow, on a dark background, with no solid-filled buttons

#### Scenario: Exact entry on a knob
- **WHEN** the user taps a knob's numeric readout
- **THEN** a text field opens for typing an exact value

### Requirement: Works on a phone and offline
The panel SHALL be usable on a phone-sized screen with touch, with tap targets of at least 44 px and no horizontal scrolling. It SHALL load all of its assets from the visualizer itself, with no requests to external hosts.

#### Scenario: Phone layout
- **WHEN** the panel is opened on a 375 px-wide screen
- **THEN** sections stack vertically, nothing scrolls horizontally, and every control can be operated by touch

#### Scenario: No internet
- **WHEN** the LAN has no internet access
- **THEN** the panel loads and works fully
