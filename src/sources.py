"""Signal sources the render loop can draw from: the generator or music files.

Every source exposes the same small interface so the main loop does not
care which one is active:

    name            "generator" or "music"
    get_buffer(dt)  (n, 2) float32 array for this frame (may be empty)
    start()         become the active source
    stop()          stop being the active source
    pause()/resume()
    is_finished()
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

EMPTY = np.zeros((0, 2), dtype=np.float32)


class Source:
    name = "source"

    def __init__(self):
        self.paused = False

    def get_buffer(self, dt: float) -> np.ndarray:
        raise NotImplementedError

    def start(self):
        self.paused = False

    def stop(self):
        pass

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def is_finished(self) -> bool:
        return False


class GeneratorSource(Source):
    """Wraps a SignalGenerator."""

    name = "generator"

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def get_buffer(self, dt: float) -> np.ndarray:
        if self.paused:
            return EMPTY
        return self.generator.get_buffer(dt)

    def reset(self):
        self.generator.reset()


class PygameMixer:
    """Thin adapter over pygame.mixer.music, initialized on first use."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.ready = False

    def _ensure(self):
        if not self.ready:
            import pygame
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2)
            self.ready = True

    def load_and_play(self, path: Path, start: float = 0.0):
        import pygame
        self._ensure()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play(start=start)

    def pause(self):
        if self.ready:
            import pygame
            pygame.mixer.music.pause()

    def unpause(self):
        if self.ready:
            import pygame
            pygame.mixer.music.unpause()

    def stop(self):
        if self.ready:
            import pygame
            pygame.mixer.music.stop()

    def get_pos_seconds(self) -> Optional[float]:
        if not self.ready:
            return None
        import pygame
        ms = pygame.mixer.music.get_pos()
        return ms / 1000.0 if ms >= 0 else None

    def quit(self):
        if self.ready:
            import pygame
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            self.ready = False


class MusicSource(Source):
    """Plays MP3s from a fixed directory and visualizes them in sync."""

    name = "music"

    def __init__(self, audio_processor, file_manager, mixer, auto_advance: bool = True):
        super().__init__()
        self.audio = audio_processor
        self.files = file_manager
        self.mixer = mixer
        self.auto_advance = auto_advance
        self.loaded = False
        self.files.scan_directory()

    # -- track list ---------------------------------------------------------

    def tracks(self) -> List[str]:
        return [p.name for p in self.files.playlist]

    def current(self) -> Optional[str]:
        if not self.loaded:
            return None
        path = self.files.get_current_file()
        return path.name if path else None

    @property
    def playing(self) -> bool:
        return self.loaded and not self.paused

    # -- transport ----------------------------------------------------------

    def start(self):
        super().start()
        if not self.loaded:
            self._load_current()
        else:
            self.mixer.unpause()

    def stop(self):
        self.mixer.stop()
        self.loaded = False

    def select(self, track: str) -> bool:
        """Make `track` current without loading it (used before start())."""
        names = self.tracks()
        if track not in names:
            return False
        self.files.current_index = names.index(track)
        self.loaded = False
        return True

    def play(self, track: str) -> bool:
        """Play a track by file name. Returns False if it is not in the list."""
        if not self.select(track):
            return False
        self.paused = False
        return self._load_current()

    def next(self) -> bool:
        if self.files.next_track() is None:
            return False
        self.paused = False
        return self._load_current()

    def prev(self) -> bool:
        if self.files.previous_track() is None:
            return False
        self.paused = False
        return self._load_current()

    def pause(self):
        super().pause()
        self.mixer.pause()

    def resume(self):
        super().resume()
        self.mixer.unpause()

    def _load_current(self) -> bool:
        """Load the current track, skipping unplayable files."""
        for _ in range(max(1, len(self.files.playlist))):
            path = self.files.get_current_file()
            if path is None:
                self.loaded = False
                return False
            print(f"Loading: {path.name}")
            self.mixer.stop()
            try:
                if not self.audio.load_file(path):
                    raise RuntimeError("could not decode audio for visualization")
                self.mixer.load_and_play(path)
                self.loaded = True
                return True
            except Exception as e:
                print(f"Error loading {path.name}: {e}")
                if self.files.next_track() is None:
                    break
        print("No playable tracks found")
        self.loaded = False
        return False

    # -- per frame ----------------------------------------------------------

    def get_buffer(self, dt: float) -> np.ndarray:
        if self.paused or not self.loaded:
            return EMPTY

        # Keep the visualization locked to what the speakers are playing
        pos = self.mixer.get_pos_seconds()
        if pos is not None:
            self.audio.seek(pos)

        buf = self.audio.get_buffer()
        if buf is None or self.audio.is_finished():
            if self.auto_advance:
                self.next()
            else:
                self.pause()
            return EMPTY
        return buf

    def is_finished(self) -> bool:
        return self.loaded and self.audio.is_finished()
