"""Signal generator module for built-in Lissajous pattern generation."""

import numpy as np
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


def parse_value(value: Any) -> Union[float, 'RandomValue']:
    """Parse a config value, which may be a number or a rand() expression.

    Args:
        value: Either a number or a string like "rand(min, max)"

    Returns:
        Either a float or a RandomValue instance
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Try to parse rand(min, max)
        match = re.match(r'rand\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)', value.strip())
        if match:
            min_val = float(match.group(1))
            max_val = float(match.group(2))
            return RandomValue(min_val, max_val)
        # Try to parse as a plain number
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Cannot parse value: {value}")

    return float(value)


class RandomValue:
    """A value that is randomly generated within a range each cycle."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val
        self.current_value = self.roll()

    def roll(self) -> float:
        """Generate a new random value."""
        self.current_value = np.random.uniform(self.min_val, self.max_val)
        return self.current_value

    def get(self) -> float:
        """Get the current value."""
        return self.current_value

    def __repr__(self):
        return f"RandomValue({self.min_val}, {self.max_val}) = {self.current_value:.3f}"


class WaveformGenerator(ABC):
    """Base class for waveform generators."""

    @abstractmethod
    def generate_sample(self, phase: float) -> float:
        """Generate a single sample at the given phase.

        Args:
            phase: Phase in radians

        Returns:
            Sample value in range [-1, 1]
        """
        pass

    def generate(self, t: np.ndarray, frequency: float, phase: float, amplitude: float) -> np.ndarray:
        """Generate waveform samples (legacy method for non-keyframed use).

        Args:
            t: Time values in seconds
            frequency: Frequency in Hz
            phase: Phase offset in radians
            amplitude: Amplitude multiplier (0.0 to 1.0)

        Returns:
            Array of sample values
        """
        phases = 2 * np.pi * frequency * t + phase
        return amplitude * np.array([self.generate_sample(p) for p in phases])


class SineGenerator(WaveformGenerator):
    """Generates sine wave."""

    def generate_sample(self, phase: float) -> float:
        return np.sin(phase)

    def generate(self, t: np.ndarray, frequency: float, phase: float, amplitude: float) -> np.ndarray:
        # Optimized vectorized version
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)


class TriangleGenerator(WaveformGenerator):
    """Generates triangle wave."""

    def generate_sample(self, phase: float) -> float:
        # Normalize phase to [0, 1)
        p = (phase / (2 * np.pi)) % 1.0
        return 4 * abs(p - 0.5) - 1

    def generate(self, t: np.ndarray, frequency: float, phase: float, amplitude: float) -> np.ndarray:
        p = (t * frequency + phase / (2 * np.pi)) % 1.0
        return amplitude * (4 * np.abs(p - 0.5) - 1)


class SawtoothGenerator(WaveformGenerator):
    """Generates sawtooth wave."""

    def generate_sample(self, phase: float) -> float:
        p = (phase / (2 * np.pi)) % 1.0
        return 2 * p - 1

    def generate(self, t: np.ndarray, frequency: float, phase: float, amplitude: float) -> np.ndarray:
        p = (t * frequency + phase / (2 * np.pi)) % 1.0
        return amplitude * (2 * p - 1)


class SquareGenerator(WaveformGenerator):
    """Generates square wave."""

    def generate_sample(self, phase: float) -> float:
        return 1.0 if np.sin(phase) >= 0 else -1.0

    def generate(self, t: np.ndarray, frequency: float, phase: float, amplitude: float) -> np.ndarray:
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase))


# Registry of available waveform generators
WAVEFORM_GENERATORS: Dict[str, WaveformGenerator] = {
    'sine': SineGenerator(),
    'triangle': TriangleGenerator(),
    'sawtooth': SawtoothGenerator(),
    'square': SquareGenerator(),
}


class Keyframe:
    """A single keyframe with frequency, phase, amplitude, and duration.

    Values can be either fixed numbers or rand(min, max) expressions that
    are re-evaluated each time the keyframe cycle loops.
    """

    def __init__(self, data: Dict[str, Any], duration_required: bool = False):
        # Parse values - may be floats or RandomValue instances
        self._frequency = parse_value(data.get('frequency', 1.0))
        self._phase = parse_value(data.get('phase', 0.0))
        self._amplitude = parse_value(data.get('amplitude', 0.8))

        if duration_required and 'duration' not in data:
            raise ValueError("Keyframe 'duration' is required when multiple keyframes are defined")
        self._duration = parse_value(data.get('duration', 1.0))

    def _get_value(self, val: Union[float, RandomValue]) -> float:
        """Get the current value, whether it's fixed or random."""
        if isinstance(val, RandomValue):
            return val.get()
        return val

    @property
    def frequency(self) -> float:
        return self._get_value(self._frequency)

    @property
    def phase(self) -> float:
        return self._get_value(self._phase)

    @property
    def amplitude(self) -> float:
        return self._get_value(self._amplitude)

    @property
    def duration(self) -> float:
        return self._get_value(self._duration)

    def reroll(self):
        """Re-roll all random values for this keyframe."""
        if isinstance(self._frequency, RandomValue):
            self._frequency.roll()
        if isinstance(self._phase, RandomValue):
            self._phase.roll()
        if isinstance(self._amplitude, RandomValue):
            self._amplitude.roll()
        if isinstance(self._duration, RandomValue):
            self._duration.roll()

    def has_randoms(self) -> bool:
        """Check if this keyframe has any random values."""
        return any(isinstance(v, RandomValue) for v in
                   [self._frequency, self._phase, self._amplitude, self._duration])

    def lerp(self, other: 'Keyframe', t: float) -> tuple:
        """Linearly interpolate between this keyframe and another.

        Args:
            other: The target keyframe
            t: Interpolation factor (0.0 = this, 1.0 = other)

        Returns:
            Tuple of (frequency, phase, amplitude)
        """
        t = max(0.0, min(1.0, t))  # Clamp to [0, 1]
        return (
            self.frequency + (other.frequency - self.frequency) * t,
            self.phase + (other.phase - self.phase) * t,
            self.amplitude + (other.amplitude - self.amplitude) * t,
        )


class AxisConfig:
    """Configuration for a single axis (X or Y) with keyframe support."""

    def __init__(self, config: Dict[str, Any]):
        self.waveform = config.get('waveform', 'sine')

        # Get the generator for this waveform
        if self.waveform not in WAVEFORM_GENERATORS:
            print(f"Unknown waveform '{self.waveform}', falling back to sine")
            self.waveform = 'sine'
        self.generator = WAVEFORM_GENERATORS[self.waveform]

        # Parse keyframes
        self.keyframes: List[Keyframe] = []
        self._parse_keyframes(config)

        # Track cycle for re-rolling randoms
        self._current_cycle = 0
        self._current_kf_index = 0
        self._update_total_duration()

    def _parse_keyframes(self, config: Dict[str, Any]):
        """Parse keyframes from config, supporting both old and new formats."""
        if 'keyframes' in config:
            # New format: array of keyframe objects
            keyframe_data = config['keyframes']
            if not isinstance(keyframe_data, list) or len(keyframe_data) == 0:
                raise ValueError("'keyframes' must be a non-empty array")

            duration_required = len(keyframe_data) > 1
            for kf_data in keyframe_data:
                self.keyframes.append(Keyframe(kf_data, duration_required))
        else:
            # Legacy format: separate arrays for each parameter
            # Convert to single keyframe
            frequency = config.get('frequency', [1.0])
            phase = config.get('phase', [0.0])
            amplitude = config.get('amplitude', [0.9])

            # Handle both scalar and array inputs
            if not isinstance(frequency, list):
                frequency = [frequency]
            if not isinstance(phase, list):
                phase = [phase]
            if not isinstance(amplitude, list):
                amplitude = [amplitude]

            self.keyframes.append(Keyframe({
                'frequency': frequency[0],
                'phase': phase[0],
                'amplitude': amplitude[0],
                'duration': 1.0
            }))

    def _update_total_duration(self):
        """Recalculate total cycle duration (needed after rerolling random durations)."""
        self.total_duration = sum(kf.duration for kf in self.keyframes)

    def reroll_all(self):
        """Re-roll all random values in all keyframes."""
        for kf in self.keyframes:
            kf.reroll()
        self._update_total_duration()

    def _find_keyframe_index(self, time: float) -> int:
        """Find the keyframe index for the segment we're interpolating FROM.

        Args:
            time: Current time in seconds

        Returns:
            Index of the current keyframe segment
        """
        if self.total_duration <= 0:
            return 0
        cycle_time = time % self.total_duration
        elapsed = 0.0
        for i, kf in enumerate(self.keyframes):
            if elapsed + kf.duration > cycle_time:
                return i
            elapsed += kf.duration
        return len(self.keyframes) - 1

    def advance(self, time: float):
        """Advance time, pre-rolling the next keyframe's randoms when segments change.

        Args:
            time: Current time in seconds
        """
        if self.total_duration <= 0:
            return

        if len(self.keyframes) == 1:
            # Single keyframe: reroll at cycle boundary (preserves original behavior)
            current_cycle = int(time / self.total_duration)
            if current_cycle > self._current_cycle:
                self._current_cycle = current_cycle
                self.reroll_all()
            return

        idx = self._find_keyframe_index(time)
        if idx != self._current_kf_index:
            self._current_kf_index = idx
            # Pre-roll the keyframe we'll interpolate toward NEXT
            preroll_idx = (idx + 2) % len(self.keyframes)
            self.keyframes[preroll_idx].reroll()
            self._update_total_duration()

    def reset(self):
        """Reset cycle tracking and reroll randoms."""
        self._current_cycle = 0
        self._current_kf_index = 0
        self.reroll_all()

    def _get_interpolated_params(self, time: float) -> tuple:
        """Get interpolated frequency, phase, amplitude at given time.

        Args:
            time: Current time in seconds

        Returns:
            Tuple of (frequency, phase, amplitude)
        """
        if len(self.keyframes) == 1:
            kf = self.keyframes[0]
            return (kf.frequency, kf.phase, kf.amplitude)

        # Find position in cycle (loop)
        cycle_time = time % self.total_duration if self.total_duration > 0 else 0

        # Find current keyframe and interpolation factor
        elapsed = 0.0
        for i, kf in enumerate(self.keyframes):
            if elapsed + kf.duration > cycle_time:
                # We're in this keyframe's duration
                t = (cycle_time - elapsed) / kf.duration if kf.duration > 0 else 0
                next_kf = self.keyframes[(i + 1) % len(self.keyframes)]
                return kf.lerp(next_kf, t)
            elapsed += kf.duration

        # Shouldn't reach here, but return last keyframe values as fallback
        kf = self.keyframes[-1]
        return (kf.frequency, kf.phase, kf.amplitude)

    def get_frequency(self, time: float = 0.0) -> float:
        """Get frequency at given time with keyframe interpolation."""
        return self._get_interpolated_params(time)[0]

    def get_phase(self, time: float = 0.0) -> float:
        """Get phase at given time with keyframe interpolation."""
        return self._get_interpolated_params(time)[1]

    def get_amplitude(self, time: float = 0.0) -> float:
        """Get amplitude at given time with keyframe interpolation."""
        return self._get_interpolated_params(time)[2]

    def generate(self, t: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Generate samples for this axis.

        Args:
            t: Time values array for sample generation
            time: Current playback time (for keyframe interpolation)

        Returns:
            Array of sample values
        """
        freq, phase, amp = self._get_interpolated_params(time)
        return self.generator.generate(t, freq, phase, amp)


class SignalGenerator:
    """Generates Lissajous patterns from configured waveforms."""

    def __init__(self, config: Dict[str, Any], sample_rate: int = 44100, buffer_size: int = 2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.enabled = config.get('enabled', False)

        # Parse X and Y axis configurations
        default_x = {
            'waveform': 'sine',
            'keyframes': [{'frequency': 3.0, 'phase': 0.0, 'amplitude': 0.8}]
        }
        default_y = {
            'waveform': 'sine',
            'keyframes': [{'frequency': 4.0, 'phase': 0.0, 'amplitude': 0.8}]
        }
        x_config = config.get('x', default_x)
        y_config = config.get('y', default_y)

        self.x_axis = AxisConfig(x_config)
        self.y_axis = AxisConfig(y_config)

        # Time tracking
        self.current_time = 0.0

        # Phase accumulators for smooth frequency transitions
        # This prevents discontinuities when frequency changes
        self.x_phase = 0.0
        self.y_phase = 0.0

    def get_buffer(self) -> np.ndarray:
        """Get the next buffer of generated samples.

        Returns:
            numpy array of shape (buffer_size, 2) with X and Y values
        """
        x_values = np.zeros(self.buffer_size, dtype=np.float32)
        y_values = np.zeros(self.buffer_size, dtype=np.float32)

        dt = 1.0 / self.sample_rate

        # Generate samples one at a time with phase accumulation
        # This ensures smooth transitions when frequency changes
        for i in range(self.buffer_size):
            sample_time = self.current_time + i * dt

            # Check for cycle boundaries and reroll randoms if needed
            self.x_axis.advance(sample_time)
            self.y_axis.advance(sample_time)

            # Get interpolated parameters at this instant
            x_freq, x_phase_offset, x_amp = self.x_axis._get_interpolated_params(sample_time)
            y_freq, y_phase_offset, y_amp = self.y_axis._get_interpolated_params(sample_time)

            # Generate samples using accumulated phase + offset
            x_values[i] = x_amp * self.x_axis.generator.generate_sample(
                self.x_phase + x_phase_offset)
            y_values[i] = y_amp * self.y_axis.generator.generate_sample(
                self.y_phase + y_phase_offset)

            # Advance phase accumulators based on instantaneous frequency
            self.x_phase += 2.0 * np.pi * x_freq * dt
            self.y_phase += 2.0 * np.pi * y_freq * dt

            # Keep phase in reasonable range to avoid float precision issues
            if self.x_phase > 2.0 * np.pi:
                self.x_phase -= 2.0 * np.pi
            if self.y_phase > 2.0 * np.pi:
                self.y_phase -= 2.0 * np.pi

        # Advance time
        self.current_time += self.buffer_size * dt

        # Combine into stereo-like format (X=left, Y=right)
        return np.column_stack([x_values, y_values]).astype(np.float32)

    def reset(self):
        """Reset the generator to the beginning."""
        self.current_time = 0.0
        self.x_phase = 0.0
        self.y_phase = 0.0
        self.x_axis.reset()
        self.y_axis.reset()

    def is_finished(self) -> bool:
        """Signal generator never finishes."""
        return False

    def get_position_seconds(self) -> float:
        """Get current position in seconds."""
        return self.current_time

    @staticmethod
    def get_available_waveforms() -> List[str]:
        """Get list of available waveform types."""
        return list(WAVEFORM_GENERATORS.keys())
