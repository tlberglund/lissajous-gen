#!/usr/bin/env python3
"""
Audio Oscilloscope Visualizer

Creates Lissajous-style patterns from stereo audio files,
using the left channel for X and right channel for Y coordinates.
"""

import argparse
import copy
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import moderngl
import numpy as np
import pygame

from audio_processor import AudioProcessor, FileManager
from config_store import DEFAULTS, DebouncedWriter, PresetStore, load_startup_config
from control import AppState, ApplyConfig, Restart, Transport, apply_config, drain
from renderer import OscilloscopeRenderer, is_raspberry_pi
from signal_generator import SignalGenerator
from sources import GeneratorSource, MusicSource, PygameMixer
from web_server import start_web_server


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class Config:
    """Application configuration, backed by a normalized config document.

    The document (see config_store.DEFAULTS) is the source of truth; the
    attributes are a convenience view used by Application.
    """

    def __init__(self, doc: Optional[Dict[str, Any]] = None):
        self.doc: Dict[str, Any] = copy.deepcopy(doc if doc is not None else DEFAULTS)
        d = self.doc

        self.input_directory = d["input_directory"]

        self.resolution = d["display"]["resolution"]
        self.fullscreen = d["display"]["fullscreen"]

        viz = d["visualization"]
        self.trace_color = tuple(viz["trace_color"])
        self.trace_width = viz["trace_width"]
        self.persistence_time = viz["persistence_time"]
        self.decay_rate = viz["decay_rate"]
        self.glow_radius = viz["glow_radius"]
        self.glow_intensity = viz["glow_intensity"]
        self.background_color = tuple(viz["background_color"])
        self.beam_sharpness = viz["beam_sharpness"]

        self.sample_rate = d["audio"]["sample_rate"]
        self.buffer_size = d["audio"]["buffer_size"]
        self.amplitude_scale = d["audio"]["amplitude_scale"]

        self.playback_mode = d["playback"]["mode"]
        self.auto_advance = d["playback"]["auto_advance"]

        self.signal_generator = d["signal_generator"]
        self.web_port = d["web"]["port"]


class Application:
    """Main application controller.

    One render loop draws from whichever source is active (generator or
    music). Remote changes arrive as commands on state.commands and are
    applied at the top of each frame, on this (the GL) thread.
    """

    def __init__(self, config: Config, state: Optional[AppState] = None,
                 writer: Optional[DebouncedWriter] = None):
        self.config = config
        self.doc = copy.deepcopy(config.doc)          # the config currently applied
        self.state = state or AppState(self.doc)
        self.writer = writer
        self.running = False
        self.restart_requested = False
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
            self.width, self.height = (int(v) for v in config.resolution)

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
        if is_raspberry_pi():
            self.ctx = moderngl.create_context(require=310)
        else:
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

        self.generator = SignalGenerator(
            config=config.signal_generator,
            sample_rate=config.sample_rate,
            buffer_size=config.buffer_size
        )
        self.generator_source = GeneratorSource(self.generator)

        # The music directory is fixed for the life of the process; the mixer
        # is only initialized the first time music actually plays.
        self.mixer = PygameMixer(config.sample_rate)
        self.music = MusicSource(
            AudioProcessor(sample_rate=config.sample_rate,
                           buffer_size=config.buffer_size,
                           amplitude_scale=config.amplitude_scale),
            FileManager(input_directory=config.input_directory,
                        playback_mode=config.playback_mode),
            self.mixer,
            auto_advance=config.auto_advance,
        )

        self.source = None
        self.switch_source("generator" if config.signal_generator.get("enabled") else "music")

        # Clock for frame timing
        self.clock = pygame.time.Clock()
        # Lower FPS on Pi for better performance
        self.target_fps = 30 if self.is_pi else 60
        self._last_snapshot = 0.0

        if self.is_pi:
            print("Running on Raspberry Pi - using optimized settings")

    # -- sources ------------------------------------------------------------

    def switch_source(self, name: str):
        """Make `name` ("generator" or "music") the active source."""
        new = self.generator_source if name == "generator" else self.music
        if new is self.source:
            return
        if self.source is not None:
            self.source.stop()
        self.source = new
        self.renderer.clear()
        new.start()
        if name == "music" and not self.music.tracks():
            print(f"No MP3 files found in {self.config.input_directory}")
        print(f"Source: {name}")
        self._publish_snapshot()

    # -- main loop ----------------------------------------------------------

    def run(self):
        """Main application loop."""
        print("Press Q or ESC to quit")
        self.running = True

        while self.running:
            dt = self.clock.tick(self.target_fps) / 1000.0

            self._handle_events()
            self._process_commands()
            if self.writer is not None:
                self.writer.tick()
            if self.restart_requested:
                break

            buffer = self.source.get_buffer(dt)
            self.renderer.render_frame(buffer, dt)
            pygame.display.flip()

            now = time.monotonic()
            if now - self._last_snapshot > 0.5:
                self._publish_snapshot()
                self._last_snapshot = now

        self._cleanup()
        if self.restart_requested:
            self._reexec()

    def _publish_snapshot(self):
        self.state.update_snapshot(
            source=self.source.name if self.source else "generator",
            fps=self.clock.get_fps() if hasattr(self, "clock") else 0.0,
            tracks=self.music.tracks(),
            current_track=self.music.current(),
            playing=self.source is self.music and self.music.playing,
        )

    # -- remote commands ----------------------------------------------------

    def _process_commands(self):
        for cmd in drain(self.state.commands):
            try:
                if isinstance(cmd, ApplyConfig):
                    self._apply(cmd.doc)
                elif isinstance(cmd, Transport):
                    self._transport(cmd)
                elif isinstance(cmd, Restart):
                    self.restart_requested = True
            except Exception as e:
                # A bad command must never take the display down
                print(f"ERROR applying {type(cmd).__name__}: {e!r}")

    def _apply(self, doc: Dict[str, Any]):
        old, self.doc = self.doc, copy.deepcopy(doc)
        apply_config(self, old, self.doc)
        if self.writer is not None:
            self.writer.update(self.doc)

    def _transport(self, cmd: Transport):
        if cmd.action == "play":
            if self.source is not self.music:
                # Playing a track implies the music source; keep config in step.
                # Select first so switching sources loads only this track.
                self.music.select(cmd.track)
                doc = copy.deepcopy(self.doc)
                doc["signal_generator"]["enabled"] = False
                self.state.set_desired(doc)
                self._apply(doc)
            else:
                self.music.play(cmd.track)
        elif self.source is self.music:
            if cmd.action == "pause":
                self.music.pause()
            elif cmd.action == "resume":
                self.music.resume()
            elif cmd.action == "next":
                self.music.next()
            elif cmd.action == "prev":
                self.music.prev()
        self._publish_snapshot()

    # -- keyboard (when one is attached) ------------------------------------

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    if self.source.paused:
                        self.source.resume()
                    else:
                        self.source.pause()

                elif event.key == pygame.K_n and self.source is self.music:
                    self.music.next()

                elif event.key == pygame.K_p and self.source is self.music:
                    self.music.prev()

                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self._adjust_persistence(0.1)

                elif event.key == pygame.K_MINUS:
                    self._adjust_persistence(-0.1)

                elif event.key == pygame.K_UP:
                    self._adjust_glow(0.1)

                elif event.key == pygame.K_DOWN:
                    self._adjust_glow(-0.1)

                elif event.key == pygame.K_r:
                    if self.source is self.generator_source:
                        self.generator.reset()
                        print("Signal generator reset")
                    elif self.music.loaded:
                        self.music.audio.reset()
                        self.mixer.load_and_play(self.music.files.get_current_file())

                elif event.key == pygame.K_c:
                    # Clear persistence buffer
                    self.renderer.clear()

                elif event.key == pygame.K_LEFTBRACKET:
                    self._adjust_beam_sharpness(-0.25)

                elif event.key == pygame.K_RIGHTBRACKET:
                    self._adjust_beam_sharpness(0.25)

            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self.renderer.resize(self.width, self.height)

    def _adjust_persistence(self, delta: float):
        """Adjust decay rate."""
        new_rate = self.renderer.decay_rate + delta
        new_rate = max(0.1, min(10.0, new_rate))
        self.renderer.set_decay_rate(new_rate)
        print(f"Decay rate: {new_rate:.1f}")

    def _adjust_glow(self, delta: float):
        """Adjust glow intensity."""
        new_intensity = self.renderer.glow_intensity + delta
        new_intensity = max(0.0, min(10.0, new_intensity))
        self.renderer.set_glow_intensity(new_intensity)
        print(f"Glow intensity: {new_intensity:.1f}")

    def _adjust_beam_sharpness(self, delta: float):
        """Adjust beam sharpness."""
        new_sharpness = self.renderer.beam_sharpness + delta
        new_sharpness = max(0.1, min(10.0, new_sharpness))
        self.renderer.set_beam_sharpness(new_sharpness)
        print(f"Beam sharpness: {new_sharpness:.1f}")

    # -- shutdown and restart -----------------------------------------------

    def _cleanup(self):
        """Clean up resources."""
        if self.writer is not None:
            self.writer.flush()
        try:
            self.mixer.quit()
        except Exception as e:
            print(f"Error stopping audio: {e!r}")
        pygame.quit()

    def _reexec(self):
        """Replace this process with a fresh copy that loads config/active.json."""
        # orig_argv keeps interpreter flags such as -u. Drop --reset: it would
        # reseed active.json from the CLI file and lose the web edits.
        argv = [a for a in sys.orig_argv if a != "--reset"]
        print("Restarting...")
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable] + argv[1:])


def lan_address() -> str:
    """Best guess at this machine's LAN IP (no packets are sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audio Oscilloscope Visualizer")
    parser.add_argument("config", nargs="?",
                        help="config file (or music directory) used to seed config/active.json")
    parser.add_argument("--reset", action="store_true",
                        help="reseed config/active.json from the config argument")
    args = parser.parse_args()

    config = Config(load_startup_config(CONFIG_DIR, args.config, args.reset))

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

    state = AppState(config.doc)
    writer = DebouncedWriter(CONFIG_DIR / "active.json")
    app = Application(config, state, writer)
    # Start serving only once the app has published its initial state
    server = start_web_server(state, PresetStore(CONFIG_DIR), config.web_port)
    if server is not None:
        print(f"  on the LAN: http://{lan_address()}:{config.web_port}/")
    app.run()


if __name__ == "__main__":
    main()
