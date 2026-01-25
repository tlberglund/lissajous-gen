#!/usr/bin/env python3
"""
Audio Oscilloscope Visualizer

Creates Lissajous-style patterns from stereo audio files,
using the left channel for X and right channel for Y coordinates.
"""

import json
import platform
import sys
from pathlib import Path
from typing import Optional

import moderngl
import numpy as np
import pygame

from audio_processor import AudioProcessor, FileManager
from renderer import OscilloscopeRenderer, is_raspberry_pi
from signal_generator import SignalGenerator


class Config:
    """Application configuration."""

    def __init__(self, config_path: Optional[str] = None):
        # Default values
        self.input_directory = "./music"
        self.resolution = "auto"
        self.fullscreen = False
        self.trace_color = (0.2, 1.0, 0.3)
        self.trace_width = 2.0
        self.persistence_time = 1.5
        self.decay_rate = 0.7
        self.glow_radius = 8
        self.glow_intensity = 1.8
        self.background_color = (0.0, 0.0, 0.0)
        self.beam_sharpness = 3.0
        self.sample_rate = 44100
        self.buffer_size = 2048
        self.amplitude_scale = 0.9
        self.playback_mode = "sequential"
        self.auto_advance = True

        # Signal generator config (Phase 4)
        self.signal_generator = {
            'enabled': False,
            'x': {
                'waveform': 'sine',
                'keyframes': [{'frequency': 3.0, 'phase': 0.0, 'amplitude': 0.8}]
            },
            'y': {
                'waveform': 'sine',
                'keyframes': [{'frequency': 4.0, 'phase': 0.0, 'amplitude': 0.8}]
            }
        }

        if config_path:
            self.load(config_path)

    def load(self, config_path: str):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)

            self.input_directory = data.get('input_directory', self.input_directory)

            display = data.get('display', {})
            self.resolution = display.get('resolution', self.resolution)
            self.fullscreen = display.get('fullscreen', self.fullscreen)

            viz = data.get('visualization', {})
            if 'trace_color' in viz:
                self.trace_color = tuple(viz['trace_color'])
            self.trace_width = viz.get('trace_width', self.trace_width)
            self.persistence_time = viz.get('persistence_time', self.persistence_time)
            self.decay_rate = viz.get('decay_rate', self.decay_rate)
            self.glow_radius = viz.get('glow_radius', self.glow_radius)
            self.glow_intensity = viz.get('glow_intensity', self.glow_intensity)
            self.beam_sharpness = viz.get('beam_sharpness', self.beam_sharpness)
            if 'background_color' in viz:
                self.background_color = tuple(viz['background_color'])

            audio = data.get('audio', {})
            self.sample_rate = audio.get('sample_rate', self.sample_rate)
            self.buffer_size = audio.get('buffer_size', self.buffer_size)
            self.amplitude_scale = audio.get('amplitude_scale', self.amplitude_scale)

            playback = data.get('playback', {})
            self.playback_mode = playback.get('mode', self.playback_mode)
            self.auto_advance = playback.get('auto_advance', self.auto_advance)

            # Signal generator config
            if 'signal_generator' in data:
                sig_gen = data['signal_generator']
                self.signal_generator['enabled'] = sig_gen.get('enabled', False)
                if 'x' in sig_gen:
                    self.signal_generator['x'] = sig_gen['x']
                if 'y' in sig_gen:
                    self.signal_generator['y'] = sig_gen['y']

            print(f"Loaded configuration from {config_path}")

        except FileNotFoundError:
            print(f"Config file not found: {config_path}, using defaults")
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}, using defaults")


class Application:
    """Main application controller."""

    def __init__(self, config: Config):
        self.config = config
        self.running = False
        self.paused = False
        self.is_pi = is_raspberry_pi()

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Audio Oscilloscope Visualizer")

        # Determine resolution
        if config.resolution == "auto":
            info = pygame.display.Info()
            if config.fullscreen:
                self.width = info.current_w
                self.height = info.current_h
            else:
                # Smaller default on Pi for performance
                self.width = 800 if self.is_pi else 1280
                self.height = 600 if self.is_pi else 720
        else:
            self.width, self.height = map(int, config.resolution.split('x'))

        # Set OpenGL attributes before creating display
        if self.is_pi:
            # Request OpenGL ES 3.0 context on Pi
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 0)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                            pygame.GL_CONTEXT_PROFILE_ES)
        else:
            # Request OpenGL 3.3 core on desktop
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK,
                                            pygame.GL_CONTEXT_PROFILE_CORE)

        # Create display
        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if config.fullscreen:
            flags |= pygame.FULLSCREEN

        pygame.display.set_mode((self.width, self.height), flags)

        # Create OpenGL context
        self.ctx = moderngl.create_context()
        print(f"OpenGL: {self.ctx.info['GL_RENDERER']}")
        print(f"Version: {self.ctx.info['GL_VERSION']}")

        # Initialize components
        self.renderer = OscilloscopeRenderer(
            ctx=self.ctx,
            width=self.width,
            height=self.height,
            trace_color=config.trace_color,
            trace_width=config.trace_width,
            decay_rate=config.decay_rate,
            glow_radius=config.glow_radius,
            glow_intensity=config.glow_intensity,
            background_color=config.background_color,
            beam_sharpness=config.beam_sharpness
        )

        self.audio_processor = AudioProcessor(
            sample_rate=config.sample_rate,
            buffer_size=config.buffer_size,
            amplitude_scale=config.amplitude_scale
        )

        self.file_manager = FileManager(
            input_directory=config.input_directory,
            playback_mode=config.playback_mode
        )

        # Signal generator (Phase 4)
        self.signal_generator = SignalGenerator(
            config=config.signal_generator,
            sample_rate=config.sample_rate,
            buffer_size=config.buffer_size
        )
        self.use_signal_generator = config.signal_generator.get('enabled', False)

        # Audio playback via pygame mixer (not needed for signal generator mode)
        if not self.use_signal_generator:
            pygame.mixer.init(frequency=config.sample_rate, size=-16, channels=2)

        # Clock for frame timing
        self.clock = pygame.time.Clock()
        # Lower FPS on Pi for better performance
        self.target_fps = 30 if self.is_pi else 60

        # Samples per frame calculation for visualization sync
        self.samples_per_frame = config.sample_rate // self.target_fps

        if self.is_pi:
            print("Running on Raspberry Pi - using optimized settings")

    def run(self):
        """Main application loop."""
        # Check if using signal generator mode
        if self.use_signal_generator:
            self._run_signal_generator_mode()
            return

        # Scan for music files
        track_count = self.file_manager.scan_directory()
        if track_count == 0:
            print(f"No MP3 files found in {self.config.input_directory}")
            print("Please add MP3 files to the music directory and restart.")
            # Run in demo mode with no audio
            self._run_demo_mode()
            return

        print(f"Found {track_count} tracks")

        # Load first track
        if not self._load_current_track():
            print("Failed to load first track")
            return

        self.running = True

        while self.running:
            dt = self.clock.tick(self.target_fps) / 1000.0  # Delta time in seconds

            # Handle events
            self._handle_events()

            if not self.paused:
                # Sync visualization to actual audio playback position
                self._sync_visualization_to_audio()

                # Get audio buffer for visualization
                audio_buffer = self.audio_processor.get_buffer()

                if audio_buffer is None or self.audio_processor.is_finished():
                    # Track finished
                    if self.config.auto_advance:
                        if not self._next_track():
                            print("Playlist finished")
                            self.running = False
                            continue
                    else:
                        self.paused = True
                        continue

                # Render frame
                self.renderer.render_frame(audio_buffer, dt)

            else:
                # When paused, still render but with no new data
                self.renderer.render_frame(np.zeros((1, 2), dtype=np.float32), dt)

            pygame.display.flip()

        self._cleanup()

    def _run_signal_generator_mode(self):
        """Run using the built-in signal generator."""
        x_cfg = self.config.signal_generator['x']
        y_cfg = self.config.signal_generator['y']
        print("Running in signal generator mode")

        # Display axis info
        for axis_name, cfg in [('X', x_cfg), ('Y', y_cfg)]:
            waveform = cfg.get('waveform', 'sine')
            keyframes = cfg.get('keyframes', [])
            if keyframes:
                n_kf = len(keyframes)
                if n_kf == 1:
                    kf = keyframes[0]
                    print(f"  {axis_name}: {waveform} @ {kf.get('frequency', 1.0)} Hz")
                else:
                    total_dur = sum(kf.get('duration', 0) for kf in keyframes)
                    print(f"  {axis_name}: {waveform}, {n_kf} keyframes, {total_dur:.1f}s cycle")

        print("Press Q or ESC to quit, R to reset")

        self.running = True

        while self.running:
            dt = self.clock.tick(self.target_fps) / 1000.0

            self._handle_events()

            if not self.paused:
                # Get buffer from signal generator
                buffer = self.signal_generator.get_buffer()
                self.renderer.render_frame(buffer, dt)
            else:
                # When paused, still render but with no new data
                self.renderer.render_frame(np.zeros((1, 2), dtype=np.float32), dt)

            pygame.display.flip()

        self._cleanup()

    def _run_demo_mode(self):
        """Run in demo mode without audio files (generates test patterns)."""
        print("Running in demo mode - generating test patterns")
        print("Press Q or ESC to quit")

        self.running = True
        time_elapsed = 0.0

        while self.running:
            dt = self.clock.tick(self.target_fps) / 1000.0
            time_elapsed += dt

            self._handle_events()

            # Generate Lissajous test pattern
            t = np.linspace(time_elapsed, time_elapsed + 0.1, self.config.buffer_size)
            freq_x = 3.0
            freq_y = 4.0
            phase = time_elapsed * 0.5

            left = np.sin(2 * np.pi * freq_x * t) * 0.8
            right = np.sin(2 * np.pi * freq_y * t + phase) * 0.8

            audio_buffer = np.column_stack([left, right]).astype(np.float32)

            self.renderer.render_frame(audio_buffer, dt)
            pygame.display.flip()

        self._cleanup()

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self._toggle_pause()

                elif event.key == pygame.K_n:
                    self._next_track()

                elif event.key == pygame.K_p:
                    self._previous_track()

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._adjust_persistence(0.1)

                elif event.key == pygame.K_MINUS:
                    self._adjust_persistence(-0.1)

                elif event.key == pygame.K_UP:
                    self._adjust_glow(0.1)

                elif event.key == pygame.K_DOWN:
                    self._adjust_glow(-0.1)

                elif event.key == pygame.K_r:
                    # Reset/restart current track or signal generator
                    if self.use_signal_generator:
                        self.signal_generator.reset()
                        print("Signal generator reset")
                    else:
                        self.audio_processor.reset()
                        self._sync_audio_playback()

                elif event.key == pygame.K_c:
                    # Clear persistence buffer
                    self.renderer.clear()

                elif event.key == pygame.K_LEFTBRACKET:
                    self._adjust_beam_sharpness(-0.5)

                elif event.key == pygame.K_RIGHTBRACKET:
                    self._adjust_beam_sharpness(0.5)

            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self.renderer.resize(self.width, self.height)

    def _load_current_track(self, skip_on_error: bool = True) -> bool:
        """Load the current track from the file manager.

        Args:
            skip_on_error: If True, automatically try the next track on failure.
        """
        max_attempts = self.file_manager.get_track_count()
        attempts = 0

        while attempts < max_attempts:
            file_path = self.file_manager.get_current_file()
            if file_path is None:
                return False

            print(f"Loading: {file_path.name}")

            # Stop any current playback
            pygame.mixer.music.stop()

            try:
                # Load for visualization
                if not self.audio_processor.load_file(file_path):
                    raise RuntimeError("Failed to decode audio for visualization")

                # Load for audio playback
                pygame.mixer.music.load(str(file_path))
                pygame.mixer.music.play()

                self.paused = False
                self.renderer.clear()
                return True

            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                attempts += 1

                if skip_on_error and attempts < max_attempts:
                    print("Skipping to next track...")
                    self.file_manager.next_track()
                else:
                    return False

        print("No playable tracks found")
        return False

    def _next_track(self) -> bool:
        """Advance to the next track."""
        next_file = self.file_manager.next_track()
        if next_file is None:
            return False
        return self._load_current_track()

    def _previous_track(self) -> bool:
        """Go to the previous track."""
        prev_file = self.file_manager.previous_track()
        if prev_file is None:
            return False
        return self._load_current_track()

    def _toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        if self.paused:
            pygame.mixer.music.pause()
            print("Paused")
        else:
            pygame.mixer.music.unpause()
            print("Resumed")

    def _sync_visualization_to_audio(self):
        """Sync visualization position to actual audio playback time."""
        try:
            # pygame.mixer.music.get_pos() returns milliseconds since playback started
            audio_pos_ms = pygame.mixer.music.get_pos()
            if audio_pos_ms >= 0:
                audio_pos_seconds = audio_pos_ms / 1000.0
                self.audio_processor.seek(audio_pos_seconds)
        except Exception:
            # If we can't get position, just let visualization run freely
            pass

    def _sync_audio_playback(self):
        """Sync audio playback with visualization position."""
        position = self.audio_processor.get_position_seconds()
        try:
            pygame.mixer.music.play(start=position)
        except Exception:
            pygame.mixer.music.play()

    def _adjust_persistence(self, delta: float):
        """Adjust decay rate."""
        new_rate = self.renderer.decay_rate + delta
        new_rate = max(0.1, min(2.0, new_rate))
        self.renderer.set_decay_rate(new_rate)
        print(f"Decay rate: {new_rate:.1f}")

    def _adjust_glow(self, delta: float):
        """Adjust glow intensity."""
        new_intensity = self.renderer.glow_intensity + delta
        new_intensity = max(0.0, min(5.0, new_intensity))
        self.renderer.set_glow_intensity(new_intensity)
        print(f"Glow intensity: {new_intensity:.1f}")

    def _adjust_beam_sharpness(self, delta: float):
        """Adjust beam sharpness."""
        new_sharpness = self.renderer.beam_sharpness + delta
        new_sharpness = max(0.5, min(10.0, new_sharpness))
        self.renderer.set_beam_sharpness(new_sharpness)
        print(f"Beam sharpness: {new_sharpness:.1f}")

    def _cleanup(self):
        """Clean up resources."""
        if not self.use_signal_generator:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except Exception:
                pass
        pygame.quit()


def main():
    """Main entry point."""
    # Look for config file
    config_path = "config.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # Check if config path is actually a directory (music folder)
    if Path(config_path).is_dir():
        config = Config()
        config.input_directory = config_path
    else:
        config = Config(config_path)

    print("Audio Oscilloscope Visualizer")
    print("=" * 40)
    print(f"Platform: {platform.system()}")
    if is_raspberry_pi():
        print("Detected: Raspberry Pi")
    if config.signal_generator.get('enabled', False):
        print("Mode: Signal Generator")
    else:
        print(f"Mode: MP3 Playback")
        print(f"Music directory: {config.input_directory}")
    print(f"Resolution: {config.resolution}")
    print(f"Fullscreen: {config.fullscreen}")
    print()
    print("Controls:")
    print("  Space     - Pause/Resume")
    print("  N         - Next track")
    print("  P         - Previous track")
    print("  R         - Restart track")
    print("  C         - Clear display")
    print("  +/-       - Adjust persistence")
    print("  Up/Down   - Adjust glow intensity")
    print("  [/]       - Adjust beam sharpness")
    print("  Q/Esc     - Quit")
    print()

    app = Application(config)
    app.run()


if __name__ == "__main__":
    main()
