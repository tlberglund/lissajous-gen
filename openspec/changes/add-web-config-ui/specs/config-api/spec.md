## Purpose

Defines the embedded HTTP server and JSON API through which a browser on the local network reads and changes the visualizer's configuration, presets, source, and music playback.

## ADDED Requirements

### Requirement: Embedded LAN server
The system SHALL run an HTTP server inside the visualizer process, bound to all interfaces on the port given by `web.port` (default 8080), without authentication. The server SHALL serve the control panel at `/` and only static files from the control panel's asset directory.

#### Scenario: Reachable from the LAN
- **WHEN** a device on the same network requests `http://<pi-address>:8080/`
- **THEN** it receives the control panel page

#### Scenario: No file access outside assets
- **WHEN** a request path attempts to escape the asset directory (for example `/../config/active.json`)
- **THEN** the server responds 404 and serves no file

### Requirement: Server failure does not affect the display
The display SHALL keep running if the server cannot start or if handling a request fails.

#### Scenario: Port already in use
- **WHEN** the configured port is unavailable at startup
- **THEN** the system logs the error and runs the display without the web interface

#### Scenario: Handler error
- **WHEN** an unexpected error occurs while handling a request
- **THEN** that request receives a 500 response, the error is logged, and the display continues uninterrupted

### Requirement: Read and write config
`GET /api/config` SHALL return the current desired config and a `restart_required` flag. `PUT /api/config` SHALL accept a complete config document; on success it SHALL respond 200 with `restart_required`, and on validation failure it SHALL respond 400 with the list of `{path, message}` errors and SHALL NOT change the running state.

#### Scenario: Successful update
- **WHEN** a client PUTs a valid document that changes `visualization.glow_intensity`
- **THEN** the response is 200 with `restart_required: false`, and a subsequent GET returns the new value

#### Scenario: Restart-required change
- **WHEN** a client PUTs a valid document that changes `display.fullscreen`
- **THEN** the response is 200 with `restart_required: true`

#### Scenario: Validation failure
- **WHEN** a client PUTs a document with an invalid field
- **THEN** the response is 400 with an error for that field's path, and GET still returns the previous config

#### Scenario: Malformed JSON
- **WHEN** a client PUTs a body that is not valid JSON
- **THEN** the response is 400 and nothing changes

### Requirement: Preset endpoints
The API SHALL provide `GET /api/presets` to list preset names, `POST /api/presets/<name>/load` to load a preset into the active config, and `POST /api/presets/<name>` with an optional `overwrite` flag to save the active config as a preset, following the config-store preset rules.

#### Scenario: List presets
- **WHEN** a client requests `GET /api/presets`
- **THEN** the response lists the preset names

#### Scenario: Save conflict
- **WHEN** a client saves to an existing preset name without `overwrite: true`
- **THEN** the response is 409 and the preset is unchanged

### Requirement: Music endpoints
The API SHALL provide `GET /api/music`, which returns the track list from the music directory, the current track, and the play state. It SHALL provide `POST /api/music/<action>` for `play` (with a track name), `pause`, `resume`, `next`, and `prev`.

#### Scenario: Track listing
- **WHEN** the music directory contains three MP3 files
- **THEN** `GET /api/music` lists those three tracks

#### Scenario: Play selected track
- **WHEN** a client posts `play` with one of the listed track names
- **THEN** the source switches to music if needed and that track starts playing

#### Scenario: Unknown track
- **WHEN** a client posts `play` with a name not in the track list
- **THEN** the response is 404 and playback is unchanged

### Requirement: Restart endpoint
`POST /api/restart` SHALL cause the visualizer to persist its active config and restart itself, coming back up with the active config.

#### Scenario: Apply restart-required settings
- **WHEN** fullscreen was changed and a client posts to `/api/restart`
- **THEN** the visualizer restarts and runs with the new fullscreen setting, and the control panel is reachable again afterward

### Requirement: Status endpoint
`GET /api/status` SHALL report the current source, frame rate, and uptime.

#### Scenario: Status read
- **WHEN** a client requests `GET /api/status` while the generator is running
- **THEN** the response reports source `generator` and a positive frame rate
