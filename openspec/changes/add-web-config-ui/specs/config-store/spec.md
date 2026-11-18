## Purpose

Defines how the visualizer persists its running configuration, validates and normalizes config documents, and manages named presets so that remote edits survive restarts and reboots.

## ADDED Requirements

### Requirement: Active config persistence
The system SHALL persist the running configuration to `config/active.json` whenever an accepted change is applied, writing at most once per second and flushing any pending write before exit or restart.

#### Scenario: Change is persisted
- **WHEN** a valid config change is accepted
- **THEN** `config/active.json` reflects that change within about one second

#### Scenario: Rapid changes are coalesced
- **WHEN** many valid changes are accepted within one second
- **THEN** the file is written no more than once in that second and ends up containing the last accepted change

#### Scenario: Pending write flushed on restart
- **WHEN** a restart is requested within one second of a change
- **THEN** the change is written to `config/active.json` before the process re-executes

### Requirement: Atomic writes
The system SHALL write config and preset files atomically so that an interrupted write never leaves a partially written file in place.

#### Scenario: Power loss during write
- **WHEN** the process is killed while writing `config/active.json`
- **THEN** the file on disk is either the previous complete version or the new complete version

### Requirement: Startup precedence
On startup, the system SHALL load `config/active.json` if it exists and is valid. If it does not exist, the system SHALL seed it from the config path given on the command line (or from defaults if none is given). A `--reset` flag SHALL force reseeding from the command-line path even when `active.json` exists.

#### Scenario: Web edits survive reboot
- **WHEN** the app starts with `config/config-generator.json` as its argument and a valid `config/active.json` exists
- **THEN** the app runs with the contents of `config/active.json`

#### Scenario: First run seeds active config
- **WHEN** the app starts with a config path argument and no `config/active.json` exists
- **THEN** `config/active.json` is created from that file and the app runs with it

#### Scenario: Reset flag
- **WHEN** the app starts with `--reset` and a config path argument
- **THEN** `config/active.json` is replaced by the normalized contents of that file

#### Scenario: Invalid active config at startup
- **WHEN** `config/active.json` exists but fails validation
- **THEN** the system logs each validation error with its field path and falls back to seeding from the command-line path or defaults, without exiting

### Requirement: Validation
The system SHALL validate every config document before it is applied or saved, and SHALL reject an invalid document as a whole, reporting each error with a dotted field path (for example `signal_generator.x.keyframes[2].frequency`) and a human-readable message.

#### Scenario: Out-of-range value
- **WHEN** a document sets a keyframe amplitude to 1.5
- **THEN** the document is rejected with an error whose path identifies that keyframe's amplitude

#### Scenario: Malformed rand expression
- **WHEN** a document sets a value to `rand(5,`
- **THEN** the document is rejected with an error at that field's path

#### Scenario: Missing duration with multiple keyframes
- **WHEN** an axis has two or more keyframes and one lacks a duration
- **THEN** the document is rejected with an error at that keyframe's duration path

#### Scenario: Unknown waveform
- **WHEN** an axis specifies a waveform other than sine, triangle, sawtooth, or square
- **THEN** the document is rejected with an error at that axis's waveform path

### Requirement: Normalization
The system SHALL normalize accepted documents before storing them: numeric strings become JSON numbers; `rand(min, max)` expressions are kept as strings in the canonical form `rand(<min>, <max>)` with min not greater than max; and the legacy per-axis `frequency`/`phase`/`amplitude` form becomes a single-element `keyframes` array.

#### Scenario: Numeric string phase
- **WHEN** a document contains `"phase": "0.78"`
- **THEN** the stored document contains `"phase": 0.78`

#### Scenario: Reversed rand bounds
- **WHEN** a document contains `"frequency": "rand(5, 1)"`
- **THEN** the stored document contains `"frequency": "rand(1, 5)"`

#### Scenario: Legacy axis form
- **WHEN** an axis is specified with top-level `frequency`, `phase`, and `amplitude` and no `keyframes`
- **THEN** the stored axis has a `keyframes` array with one keyframe holding those values

### Requirement: Presets
The system SHALL treat every `config/*.json` file other than `active.json` as a named preset (name = filename without extension). It SHALL support listing presets, loading a preset into the active config, and saving the active config as a preset. Preset names SHALL match `[A-Za-z0-9_-]+`, and the name `active` SHALL be reserved.

#### Scenario: Existing files are presets
- **WHEN** presets are listed with the repository's shipped config files present
- **THEN** the list includes `config-generator`, `config-generator-keyframed`, `config-generator-random`, and `config-music`, and does not include `active`

#### Scenario: Load preset
- **WHEN** the `config-generator-keyframed` preset is loaded
- **THEN** its normalized contents become the active config and are applied as a normal config change

#### Scenario: Save new preset
- **WHEN** the active config is saved as preset `spirals`
- **THEN** `config/spirals.json` is created containing the active config

#### Scenario: Refuse overwrite without confirmation
- **WHEN** a save is requested to an existing preset name without an overwrite confirmation
- **THEN** the save is rejected and the existing file is unchanged

#### Scenario: Invalid preset name
- **WHEN** a save or load names a preset such as `../etc/passwd` or `active`
- **THEN** the request is rejected and no file outside the preset set is read or written
