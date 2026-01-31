"""Audio processing module for MP3 decoding and channel extraction."""

import numpy as np
from pydub import AudioSegment
from pathlib import Path
from typing import Optional, Tuple
import threading
import queue


class AudioProcessor:
    """Handles MP3 decoding and provides stereo channel data for visualization."""

    def __init__(self, sample_rate: int = 44100, buffer_size: int = 2048,
                 amplitude_scale: float = 0.9):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.amplitude_scale = amplitude_scale

        self.samples: Optional[np.ndarray] = None  # Shape: (num_samples, 2)
        self.current_position = 0
        self.is_loaded = False
        self.current_file: Optional[Path] = None
        self.duration_seconds = 0.0

    def load_file(self, file_path: Path) -> bool:
        """Load and decode an MP3 file."""
        try:
            audio = AudioSegment.from_mp3(str(file_path))

            # Convert to target sample rate if needed
            if audio.frame_rate != self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)

            # Ensure stereo
            if audio.channels == 1:
                audio = audio.set_channels(2)

            # Get raw samples as numpy array
            raw_samples = np.array(audio.get_array_of_samples())

            # Determine bit depth for normalization
            if audio.sample_width == 1:
                max_val = 128.0
            elif audio.sample_width == 2:
                max_val = 32768.0
            elif audio.sample_width == 4:
                max_val = 2147483648.0
            else:
                max_val = 32768.0

            # Normalize to [-1, 1]
            raw_samples = raw_samples.astype(np.float32) / max_val

            # Reshape to stereo (interleaved L, R, L, R, ...)
            self.samples = raw_samples.reshape(-1, 2)

            # Apply amplitude scaling
            self.samples *= self.amplitude_scale

            self.current_position = 0
            self.is_loaded = True
            self.current_file = file_path
            self.duration_seconds = len(self.samples) / self.sample_rate

            return True

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            self.is_loaded = False
            return False

    def get_buffer(self) -> Optional[np.ndarray]:
        """Get the next buffer of samples for visualization.

        Returns:
            numpy array of shape (buffer_size, 2) with left and right channels,
            or None if no more data.
        """
        if not self.is_loaded or self.samples is None:
            return None

        end_position = self.current_position + self.buffer_size

        if self.current_position >= len(self.samples):
            return None

        # Handle case where we're near the end
        if end_position > len(self.samples):
            # Pad with zeros if needed
            remaining = self.samples[self.current_position:]
            padding = np.zeros((end_position - len(self.samples), 2), dtype=np.float32)
            buffer = np.vstack([remaining, padding])
        else:
            buffer = self.samples[self.current_position:end_position]

        self.current_position = end_position
        return buffer.astype(np.float32)

    def get_position_seconds(self) -> float:
        """Get current playback position in seconds."""
        if not self.is_loaded:
            return 0.0
        return self.current_position / self.sample_rate

    def get_progress(self) -> float:
        """Get playback progress as fraction 0.0-1.0."""
        if not self.is_loaded or self.samples is None:
            return 0.0
        return min(1.0, self.current_position / len(self.samples))

    def is_finished(self) -> bool:
        """Check if we've reached the end of the current track."""
        if not self.is_loaded or self.samples is None:
            return True
        return self.current_position >= len(self.samples)

    def reset(self):
        """Reset playback position to the beginning."""
        self.current_position = 0

    def seek(self, position_seconds: float):
        """Seek to a specific position in seconds."""
        if self.is_loaded:
            self.current_position = int(position_seconds * self.sample_rate)
            self.current_position = max(0, min(self.current_position,
                                               len(self.samples) - 1 if self.samples is not None else 0))


class FileManager:
    """Manages MP3 file discovery and playlist."""

    def __init__(self, input_directory: str, playback_mode: str = "sequential"):
        self.input_directory = Path(input_directory)
        self.playback_mode = playback_mode
        self.playlist: list[Path] = []
        self.current_index = 0

    def scan_directory(self) -> int:
        """Scan the input directory for MP3 files.

        Returns:
            Number of files found.
        """
        self.playlist = []

        if not self.input_directory.exists():
            print(f"Directory not found: {self.input_directory}")
            return 0

        # Find all MP3 files
        for pattern in ['*.mp3', '*.MP3']:
            self.playlist.extend(self.input_directory.glob(pattern))

        # Sort alphabetically for sequential mode
        self.playlist.sort(key=lambda p: p.name.lower())

        if self.playback_mode == "random":
            import random
            random.shuffle(self.playlist)

        self.current_index = 0
        return len(self.playlist)

    def get_current_file(self) -> Optional[Path]:
        """Get the current file in the playlist."""
        if not self.playlist or self.current_index >= len(self.playlist):
            return None
        return self.playlist[self.current_index]

    def next_track(self) -> Optional[Path]:
        """Advance to the next track."""
        if not self.playlist:
            return None

        if self.playback_mode == "single-loop":
            return self.playlist[self.current_index]

        self.current_index += 1

        if self.current_index >= len(self.playlist):
            if self.playback_mode == "sequential":
                self.current_index = 0  # Loop back to start
            else:
                return None

        return self.playlist[self.current_index]

    def previous_track(self) -> Optional[Path]:
        """Go to the previous track."""
        if not self.playlist:
            return None

        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.playlist) - 1

        return self.playlist[self.current_index]

    def get_track_count(self) -> int:
        """Get total number of tracks in playlist."""
        return len(self.playlist)

    def get_current_index(self) -> int:
        """Get current track index (0-based)."""
        return self.current_index
