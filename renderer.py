"""OpenGL renderer for oscilloscope visualization with persistence and glow effects."""

import moderngl
import numpy as np
import platform
import sys
from typing import Tuple, Optional


def is_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi."""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            return 'raspberry pi' in f.read().lower()
    except:
        return False


def get_gl_version() -> str:
    """Get appropriate GLSL version directive based on platform."""
    if is_raspberry_pi() or platform.system() == 'Linux':
        # Try OpenGL ES 3.0 on Linux/Pi, fall back handled by moderngl
        return '#version 300 es'
    else:
        # macOS and Windows use desktop OpenGL
        return '#version 330'


def get_precision() -> str:
    """Get precision qualifier for ES (empty for desktop GL)."""
    if is_raspberry_pi() or platform.system() == 'Linux':
        return 'precision highp float;'
    return ''


# Generate shaders with appropriate version
GL_VERSION = get_gl_version()
PRECISION = get_precision()

# Shader sources
VERTEX_SHADER = f"""
{GL_VERSION}
{PRECISION}

in vec2 position;

void main() {{
    gl_Position = vec4(position, 0.0, 1.0);
}}
"""

FRAGMENT_SHADER = f"""
{GL_VERSION}
{PRECISION}

uniform vec3 trace_color;
out vec4 frag_color;

void main() {{
    frag_color = vec4(trace_color, 1.0);
}}
"""

FULLSCREEN_VERTEX_SHADER = f"""
{GL_VERSION}
{PRECISION}

in vec2 position;
in vec2 texcoord;
out vec2 v_texcoord;

void main() {{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}}
"""

DECAY_FRAGMENT_SHADER = f"""
{GL_VERSION}
{PRECISION}

uniform sampler2D texture0;
uniform float decay_factor;
in vec2 v_texcoord;
out vec4 frag_color;

void main() {{
    vec4 color = texture(texture0, v_texcoord);
    frag_color = color * decay_factor;
}}
"""

BLUR_HORIZONTAL_SHADER = f"""
{GL_VERSION}
{PRECISION}

uniform sampler2D texture0;
uniform float blur_radius;
uniform vec2 texture_size;
in vec2 v_texcoord;
out vec4 frag_color;

void main() {{
    vec4 color = vec4(0.0);
    float total_weight = 0.0;
    float pixel_size = 1.0 / texture_size.x;

    for (float i = -blur_radius; i <= blur_radius; i += 1.0) {{
        float weight = exp(-0.5 * (i * i) / (blur_radius * blur_radius * 0.25));
        color += texture(texture0, v_texcoord + vec2(i * pixel_size, 0.0)) * weight;
        total_weight += weight;
    }}

    frag_color = color / total_weight;
}}
"""

BLUR_VERTICAL_SHADER = f"""
{GL_VERSION}
{PRECISION}

uniform sampler2D texture0;
uniform float blur_radius;
uniform vec2 texture_size;
in vec2 v_texcoord;
out vec4 frag_color;

void main() {{
    vec4 color = vec4(0.0);
    float total_weight = 0.0;
    float pixel_size = 1.0 / texture_size.y;

    for (float i = -blur_radius; i <= blur_radius; i += 1.0) {{
        float weight = exp(-0.5 * (i * i) / (blur_radius * blur_radius * 0.25));
        color += texture(texture0, v_texcoord + vec2(0.0, i * pixel_size)) * weight;
        total_weight += weight;
    }}

    frag_color = color / total_weight;
}}
"""

COMBINE_SHADER = f"""
{GL_VERSION}
{PRECISION}

uniform sampler2D original_texture;
uniform sampler2D glow_texture;
uniform float glow_intensity;
uniform vec3 background_color;
in vec2 v_texcoord;
out vec4 frag_color;

void main() {{
    vec4 original = texture(original_texture, v_texcoord);
    vec4 glow = texture(glow_texture, v_texcoord);

    vec3 combined = background_color + original.rgb + glow.rgb * glow_intensity;
    frag_color = vec4(combined, 1.0);
}}
"""


class OscilloscopeRenderer:
    """Renders oscilloscope visualization with persistence and glow effects."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int,
                 trace_color: Tuple[float, float, float] = (0.2, 1.0, 0.3),
                 trace_width: float = 2.0,
                 decay_rate: float = 0.7,
                 glow_radius: int = 8,
                 glow_intensity: float = 1.8,
                 background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)):

        self.ctx = ctx
        self.width = width
        self.height = height
        self.trace_color = trace_color
        self.trace_width = trace_width
        self.decay_rate = decay_rate
        self.glow_radius = glow_radius
        self.glow_intensity = glow_intensity
        self.background_color = background_color

        # Create shader programs
        self._create_shaders()

        # Create framebuffers and textures
        self._create_framebuffers()

        # Create fullscreen quad for post-processing
        self._create_fullscreen_quad()

        # Create vertex buffer for trace (will be updated each frame)
        self.trace_vbo = self.ctx.buffer(reserve=4096 * 2 * 4)  # Reserve space for 4096 vertices
        self.trace_vao = self.ctx.vertex_array(
            self.trace_program,
            [(self.trace_vbo, '2f', 'position')]
        )

    def _create_shaders(self):
        """Create all shader programs."""
        # Trace drawing shader
        self.trace_program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )

        # Decay shader for persistence effect
        self.decay_program = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=DECAY_FRAGMENT_SHADER
        )

        # Horizontal blur shader
        self.blur_h_program = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=BLUR_HORIZONTAL_SHADER
        )

        # Vertical blur shader
        self.blur_v_program = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=BLUR_VERTICAL_SHADER
        )

        # Final combine shader
        self.combine_program = self.ctx.program(
            vertex_shader=FULLSCREEN_VERTEX_SHADER,
            fragment_shader=COMBINE_SHADER
        )

    def _create_texture(self, size: Tuple[int, int], components: int = 4):
        """Create a texture with appropriate format for the platform."""
        # Try half-float first (better quality), fall back to full float or unsigned byte
        for dtype in ['f2', 'f4', 'f1']:
            try:
                tex = self.ctx.texture(size, components, dtype=dtype)
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                return tex
            except Exception:
                continue
        # Last resort: default texture
        tex = self.ctx.texture(size, components)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    def _create_framebuffers(self):
        """Create framebuffers and textures for rendering pipeline."""
        # Main persistence texture (accumulates trace over time)
        self.persistence_texture = self._create_texture((self.width, self.height))
        self.persistence_fbo = self.ctx.framebuffer(color_attachments=[self.persistence_texture])

        # Temporary texture for decay operation
        self.temp_texture = self._create_texture((self.width, self.height))
        self.temp_fbo = self.ctx.framebuffer(color_attachments=[self.temp_texture])

        # Blur textures (can be smaller for performance)
        blur_width = self.width // 2
        blur_height = self.height // 2

        self.blur_texture_1 = self._create_texture((blur_width, blur_height))
        self.blur_fbo_1 = self.ctx.framebuffer(color_attachments=[self.blur_texture_1])

        self.blur_texture_2 = self._create_texture((blur_width, blur_height))
        self.blur_fbo_2 = self.ctx.framebuffer(color_attachments=[self.blur_texture_2])

        # Clear all framebuffers
        self.persistence_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.temp_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.blur_fbo_1.clear(0.0, 0.0, 0.0, 0.0)
        self.blur_fbo_2.clear(0.0, 0.0, 0.0, 0.0)

    def _create_fullscreen_quad(self):
        """Create a fullscreen quad for post-processing passes."""
        # Vertices: position (x, y) and texcoord (u, v)
        vertices = np.array([
            # position    texcoord
            -1.0, -1.0,   0.0, 0.0,
             1.0, -1.0,   1.0, 0.0,
            -1.0,  1.0,   0.0, 1.0,
             1.0,  1.0,   1.0, 1.0,
        ], dtype='f4')

        self.quad_vbo = self.ctx.buffer(vertices)

        # Create VAOs for each program
        self.decay_vao = self.ctx.vertex_array(
            self.decay_program,
            [(self.quad_vbo, '2f 2f', 'position', 'texcoord')]
        )

        self.blur_h_vao = self.ctx.vertex_array(
            self.blur_h_program,
            [(self.quad_vbo, '2f 2f', 'position', 'texcoord')]
        )

        self.blur_v_vao = self.ctx.vertex_array(
            self.blur_v_program,
            [(self.quad_vbo, '2f 2f', 'position', 'texcoord')]
        )

        self.combine_vao = self.ctx.vertex_array(
            self.combine_program,
            [(self.quad_vbo, '2f 2f', 'position', 'texcoord')]
        )

    def render_frame(self, audio_buffer: np.ndarray, dt: float):
        """Render a single frame of the oscilloscope.

        Args:
            audio_buffer: numpy array of shape (N, 2) with left/right channels
            dt: delta time since last frame in seconds
        """
        # Step 1: Apply decay to persistence texture
        self._apply_decay(dt)

        # Step 2: Draw new trace onto persistence texture
        self._draw_trace(audio_buffer)

        # Step 3: Apply glow effect
        self._apply_glow()

        # Step 4: Combine and render to screen
        self._render_final()

    def _apply_decay(self, dt: float):
        """Apply decay effect to the persistence texture."""
        # Calculate decay factor based on time
        decay_factor = 1.0 - (self.decay_rate * dt)
        decay_factor = max(0.0, min(1.0, decay_factor))

        # Render decayed persistence to temp texture
        self.temp_fbo.use()
        self.temp_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.persistence_texture.use(0)
        self.decay_program['texture0'].value = 0
        self.decay_program['decay_factor'].value = decay_factor

        self.decay_vao.render(moderngl.TRIANGLE_STRIP)

        # Copy temp back to persistence
        self.persistence_fbo.use()
        self.persistence_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.temp_texture.use(0)
        self.decay_program['texture0'].value = 0
        self.decay_program['decay_factor'].value = 1.0

        self.decay_vao.render(moderngl.TRIANGLE_STRIP)

    def _draw_trace(self, audio_buffer: np.ndarray):
        """Draw the audio trace onto the persistence texture."""
        if audio_buffer is None or len(audio_buffer) == 0:
            return

        # Prepare vertex data: left channel -> X, right channel -> Y
        vertices = audio_buffer.astype('f4').flatten()

        # Update vertex buffer
        self.trace_vbo.write(vertices.tobytes())

        # Draw to persistence texture with additive blending
        self.persistence_fbo.use()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)  # Additive blending

        self.trace_program['trace_color'].value = self.trace_color
        self.ctx.line_width = self.trace_width

        self.trace_vao.render(moderngl.LINE_STRIP, vertices=len(audio_buffer))

        self.ctx.disable(moderngl.BLEND)

    def _apply_glow(self):
        """Apply two-pass Gaussian blur for glow effect."""
        blur_width = self.width // 2
        blur_height = self.height // 2

        # Pass 1: Horizontal blur
        self.blur_fbo_1.use()
        self.blur_fbo_1.clear(0.0, 0.0, 0.0, 0.0)

        self.persistence_texture.use(0)
        self.blur_h_program['texture0'].value = 0
        self.blur_h_program['blur_radius'].value = float(self.glow_radius)
        self.blur_h_program['texture_size'].value = (float(blur_width), float(blur_height))

        self.blur_h_vao.render(moderngl.TRIANGLE_STRIP)

        # Pass 2: Vertical blur
        self.blur_fbo_2.use()
        self.blur_fbo_2.clear(0.0, 0.0, 0.0, 0.0)

        self.blur_texture_1.use(0)
        self.blur_v_program['texture0'].value = 0
        self.blur_v_program['blur_radius'].value = float(self.glow_radius)
        self.blur_v_program['texture_size'].value = (float(blur_width), float(blur_height))

        self.blur_v_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_final(self):
        """Combine original and glow textures and render to screen."""
        self.ctx.screen.use()
        self.ctx.clear(*self.background_color)

        self.persistence_texture.use(0)
        self.blur_texture_2.use(1)

        self.combine_program['original_texture'].value = 0
        self.combine_program['glow_texture'].value = 1
        self.combine_program['glow_intensity'].value = self.glow_intensity
        self.combine_program['background_color'].value = self.background_color

        self.combine_vao.render(moderngl.TRIANGLE_STRIP)

    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.width = width
        self.height = height

        # Recreate framebuffers with new size
        self._create_framebuffers()

    def clear(self):
        """Clear the persistence buffer."""
        self.persistence_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.temp_fbo.clear(0.0, 0.0, 0.0, 0.0)

    def set_trace_color(self, color: Tuple[float, float, float]):
        """Set the trace color."""
        self.trace_color = color

    def set_decay_rate(self, rate: float):
        """Set the decay rate."""
        self.decay_rate = max(0.0, min(1.0, rate))

    def set_glow_intensity(self, intensity: float):
        """Set the glow intensity."""
        self.glow_intensity = max(0.0, intensity)

    def set_glow_radius(self, radius: int):
        """Set the glow blur radius."""
        self.glow_radius = max(1, radius)
