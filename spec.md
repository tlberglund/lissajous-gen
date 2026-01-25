# Audio Oscilloscope Visualizer - Application Specification

## Technology Stack

**Recommended: Python with OpenGL**
- **Language**: Python 3.9+
- **Graphics**: OpenGL via `moderngl` or `PyOpenGL`
- **Audio Processing**: `pydub` or `librosa` for MP3 decoding, `numpy` for signal processing
- **Windowing**: `pygame` or `glfw` for window management and fullscreen display

**Alternative: C++ (for better performance)**
- **Language**: C++17
- **Graphics**: OpenGL 3.3+ or SDL2
- **Audio Processing**: `libmpg123` for MP3 decoding, JUCE framework as all-in-one option

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         Application Controller          │
│  - File discovery & playlist management │
│  - Playback state machine               │
└───────────┬─────────────────────────────┘
            │
    ┌───────┴────────┐
    ▼                ▼
┌─────────────┐  ┌──────────────────┐
│   Audio     │  │    Renderer      │
│  Processor  │  │   (OpenGL)       │
│             │  │                  │
│ - Decode MP3│  │ - Trace drawing  │
│ - Buffer    │  │ - Persistence    │
│ - L/R split │  │ - Glow effect    │
└─────────────┘  └──────────────────┘
```

---

## Core Components

### 1. File Manager
**Responsibilities:**
- Scan input directory for MP3 files
- Build and maintain playlist
- Handle file selection (sequential, random, etc.)

**Configuration:**
- `input_directory`: Path to MP3 folder
- `playback_mode`: sequential, random, single-loop

### 2. Audio Processor
**Responsibilities:**
- Decode MP3 to raw PCM samples
- Split stereo into left (X) and right (Y) channels
- Buffer samples for real-time visualization
- Resample if needed for consistent frame rate

**Key Parameters:**
- `sample_rate`: Target sample rate (44100 Hz recommended)
- `buffer_size`: Samples per render frame (e.g., 1024-4096)
- `amplitude_scale`: Normalization factor for visualization

**Processing Pipeline:**
1. Decode MP3 → stereo float samples [-1.0, 1.0]
2. Split into left/right channels
3. Buffer samples for frame-synchronized rendering
4. Map to screen coordinates: left → X position, right → Y position

### 3. Oscilloscope Renderer
**Responsibilities:**
- Full-screen OpenGL context management
- Draw X-Y trace from audio samples
- Implement trace persistence (fade/decay)
- Apply glow post-processing effect

**Rendering Technique:**

**Option A: Additive Blending with Decay**
- Render to texture with additive blending
- Apply fade/decay shader each frame
- Draw current frame's samples on top
- Glow via Gaussian blur post-process

**Option B: Trail Buffer**
- Maintain ring buffer of last N frames
- Render each with decreasing alpha
- Current frame at full brightness

**Key Visual Parameters:**
- `trace_color`: RGB color of the trace (e.g., green phosphor)
- `trace_width`: Line thickness in pixels (1-4)
- `persistence_time`: How long trace lingers (0.0-5.0 seconds)
- `decay_rate`: How quickly trace fades (0.0-1.0 per second)
- `glow_radius`: Gaussian blur radius for glow (2-20 pixels)
- `glow_intensity`: Brightness multiplier for glow (1.0-3.0)
- `background_color`: Screen background (typically black)

---

## Rendering Pipeline

### Per-Frame Process:
1. **Fetch Audio Buffer**: Get next `buffer_size` samples from current MP3
2. **Generate Vertex Data**: Create line strip from (left[i], right[i]) pairs
3. **Coordinate Mapping**: Map audio samples [-1, 1] to screen space [-1, 1] (NDC)
4. **Render to Persistence Texture**:
   - Apply decay: multiply existing texture by `(1 - decay_rate * dt)`
   - Draw new line strip with additive blending
5. **Glow Pass**: 
   - Two-pass Gaussian blur on persistence texture
   - Combine original + blurred for glow effect
6. **Display**: Render final result to screen

### Shaders:

**Vertex Shader** (simple passthrough for line drawing)
```glsl
#version 330
in vec2 position;  // X=left channel, Y=right channel
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
```

**Fragment Shader** (with glow color)
```glsl
#version 330
uniform vec3 trace_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(trace_color, 1.0);
}
```

---

## Configuration File

**Format**: JSON or TOML

```json
{
  "input_directory": "/home/pi/music",
  "display": {
    "resolution": "auto",
    "fullscreen": true
  },
  "visualization": {
    "trace_color": [0.2, 1.0, 0.3],
    "trace_width": 2.0,
    "persistence_time": 1.5,
    "decay_rate": 0.7,
    "glow_radius": 8,
    "glow_intensity": 1.8,
    "background_color": [0.0, 0.0, 0.0]
  },
  "audio": {
    "sample_rate": 44100,
    "buffer_size": 2048,
    "amplitude_scale": 0.9
  },
  "playback": {
    "mode": "sequential",
    "auto_advance": true
  }
}
```

---

## Performance Considerations

### Raspberry Pi Optimization:
- **Use GPU acceleration**: OpenGL ES 2.0/3.0 is well-supported on Pi 4/5
- **Limit sample rate**: 44.1kHz is sufficient; avoid 96kHz/192kHz
- **Optimize blur**: Use separable Gaussian blur (horizontal + vertical passes)
- **Buffer management**: Double-buffer audio to prevent stuttering
- **Target 30-60 FPS**: Balance smoothness with Pi's GPU capabilities

### Memory Usage:
- Pre-allocate vertex buffers
- Reuse framebuffer textures
- Stream MP3 decoding (don't load entire file)

---

## Input Controls (Optional)

If keyboard/remote input is available:
- **Space**: Pause/Resume visualization
- **N**: Next track
- **P**: Previous track
- **Q/Esc**: Quit application
- **+/-**: Adjust persistence time
- **Arrow Keys**: Adjust glow intensity

---

## Implementation Phases

### Phase 1: Core Functionality
- File scanning and MP3 decoding
- Basic X-Y line rendering (no effects)
- Full-screen window management

### Phase 2: Visual Effects
- Persistence/decay system
- Glow post-processing
- Color configuration

### Phase 3: Polish
- Configuration file loading
- Playlist management
- Performance optimization for Pi

---

## Example Libraries (Python)

```python
# Install:
# pip install moderngl pygame pydub numpy pillow

import moderngl
import pygame
from pydub import AudioSegment
import numpy as np
```

---

This spec provides a complete foundation for the visualizer. The implementation can be adapted based on specific performance requirements and visual preferences for the Raspberry Pi deployment.

