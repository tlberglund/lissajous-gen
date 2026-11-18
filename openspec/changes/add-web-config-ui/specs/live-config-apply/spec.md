## Purpose

Defines how accepted configuration changes take effect on the running display: which settings apply immediately, which require a restart, and how the input source is switched at runtime.

## ADDED Requirements

### Requirement: Live application of accepted changes
An accepted config change SHALL take effect on the display within one rendered frame, without restarting the process, for every setting except those listed as restart-required.

#### Scenario: Visualization change
- **WHEN** an accepted change sets `visualization.trace_color` to `[1.0, 0.5, 0.0]`
- **THEN** the trace is drawn orange from the next frame on

#### Scenario: Previously constructor-only settings
- **WHEN** an accepted change sets `visualization.trace_width` or `visualization.background_color`
- **THEN** the display reflects the new value without a restart

#### Scenario: Generator change
- **WHEN** an accepted change edits a keyframe on the X axis
- **THEN** the X-axis signal begins transitioning to the new keyframes on the next frame

### Requirement: Rapid changes are coalesced
When several accepted changes arrive between two frames, the system SHALL apply only the most recent one.

#### Scenario: Fast knob drag
- **WHEN** ten changes are accepted between two frames
- **THEN** the next frame reflects only the tenth change

### Requirement: Restart-required settings
Changes to `display.resolution`, `display.fullscreen`, `audio.sample_rate`, `audio.buffer_size`, and `web.port` SHALL be persisted but SHALL NOT take effect until a restart. The system SHALL report that a restart is pending until one occurs.

#### Scenario: Pending restart reported
- **WHEN** `display.resolution` is changed
- **THEN** the display keeps its current resolution and the API reports `restart_required: true`

#### Scenario: Restart applies setting
- **WHEN** a restart occurs after `display.resolution` was changed
- **THEN** the display comes back at the new resolution and `restart_required` is false

### Requirement: Self restart
A restart SHALL re-execute the visualizer process in place with its original arguments, so it works both under systemd and when started by hand.

#### Scenario: Restart under systemd
- **WHEN** a restart is requested on the Pi running under systemd
- **THEN** the visualizer comes back up without systemd treating it as a failure

### Requirement: Runtime source switching
The system SHALL switch between the signal generator and music playback at runtime when `signal_generator.enabled` changes or a music track is played, and SHALL clear the persistence trail on each switch.

#### Scenario: Generator to music
- **WHEN** the source is switched from generator to music
- **THEN** music playback starts with audio output, the display shows the music's trace, and no trail from the generator figure remains

#### Scenario: Music to generator
- **WHEN** the source is switched from music to generator
- **THEN** music audio stops and the generator figure is displayed

#### Scenario: Empty music directory
- **WHEN** the source is switched to music and the music directory has no playable files
- **THEN** the display keeps running (showing no trace) and the API reports an empty track list

### Requirement: Fixed music directory
The music directory SHALL be the `input_directory` in effect at startup and SHALL NOT be changeable through the web interface.

#### Scenario: Directory not editable
- **WHEN** a PUT changes `input_directory`
- **THEN** the change is rejected with an error at `input_directory`
