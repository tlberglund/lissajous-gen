## Purpose

Defines the timing and continuity behavior of the built-in signal generator: keyframe durations in real seconds, smooth glide transitions when its configuration changes, and an uninterrupted beam across changes.

## ADDED Requirements

### Requirement: Wall-clock signal time
Signal time SHALL advance at the same rate as real time, regardless of frame rate, sample rate, or buffer size, so that keyframe durations are measured in wall-clock seconds.

#### Scenario: Keyframe cycle length on Pi
- **WHEN** an axis has keyframes whose durations sum to 30 seconds and the display runs at 30 fps
- **THEN** one full keyframe cycle takes 30 seconds of wall-clock time (within 2%)

#### Scenario: Same timing on a faster display
- **WHEN** the same config runs at 60 fps
- **THEN** one full keyframe cycle still takes 30 seconds of wall-clock time (within 2%)

#### Scenario: Frame stall
- **WHEN** a single frame takes much longer than normal
- **THEN** the generator produces at most `audio.buffer_size` samples for that frame

### Requirement: Glide transitions
When the generator's configuration changes, each axis SHALL transition from its current instantaneous frequency, phase offset, and amplitude to the new configuration's first keyframe over `signal_generator.transition_time` seconds (default 1.5), and SHALL then loop the new keyframes normally. A transition time of 0 SHALL apply the new parameters immediately.

#### Scenario: Frequency glide
- **WHEN** the X axis is at a constant 3 Hz, transition time is 2 seconds, and a change sets it to 5 Hz
- **THEN** the X frequency rises continuously from 3 Hz to 5 Hz over 2 seconds and stays at 5 Hz afterward

#### Scenario: Snap
- **WHEN** transition time is 0 and a change sets the X frequency from 3 Hz to 5 Hz
- **THEN** the X frequency is 5 Hz starting with the next frame

#### Scenario: Change during a glide
- **WHEN** a second change arrives while a glide is in progress
- **THEN** the new glide starts from the mid-glide values at that moment, with no jump

#### Scenario: Glide into a keyframe cycle
- **WHEN** a change replaces a single-keyframe axis with three keyframes
- **THEN** the axis glides to the first new keyframe and then plays the three-keyframe loop from its start

### Requirement: Waveform changes snap
A change to an axis's waveform type SHALL take effect immediately on that axis without a glide. The other axis SHALL still glide if its parameters changed.

#### Scenario: Sine to square
- **WHEN** the Y axis waveform changes from sine to square while the X frequency also changes
- **THEN** Y switches to square on the next frame and X glides to its new frequency

### Requirement: Beam continuity
Applying a generator configuration change SHALL NOT reset the oscillator phase of either axis, so the plotted point moves continuously across the change (except for waveform-type snaps and instantaneous phase-offset changes when transition time is 0).

#### Scenario: No jump on frequency change
- **WHEN** a frequency change is applied with a nonzero transition time
- **THEN** consecutive output samples across the change differ by no more than they do during steady-state generation at the higher of the two frequencies

### Requirement: Transition time setting
The config SHALL accept `signal_generator.transition_time` as a non-negative number of seconds. A change to transition time SHALL itself apply immediately, without a glide.

#### Scenario: Negative transition time
- **WHEN** a document sets `transition_time` to -1
- **THEN** the document is rejected with an error at `signal_generator.transition_time`
