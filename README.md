⏺ The code is now cross-platform. Here's a summary of the changes:

  Key differences handled:
  ┌────────────────────┬──────────────┬─────────────────────────────┐
  │      Feature       │    macOS     │        Raspberry Pi         │
  ├────────────────────┼──────────────┼─────────────────────────────┤
  │ OpenGL             │ 3.3 Core     │ ES 3.0                      │
  ├────────────────────┼──────────────┼─────────────────────────────┤
  │ GLSL               │ #version 330 │ #version 300 es + precision │
  ├────────────────────┼──────────────┼─────────────────────────────┤
  │ Default resolution │ 1280x720     │ 800x600                     │
  ├────────────────────┼──────────────┼─────────────────────────────┤
  │ Target FPS         │ 60           │ 30                          │
  ├────────────────────┼──────────────┼─────────────────────────────┤
  │ Glow radius        │ 8            │ 4 (in pi config)            │
  └────────────────────┴──────────────┴─────────────────────────────┘
  On macOS - Just run as-is:
  pip install -r requirements.txt
  python main.py

  On Raspberry Pi - You'll also need:
  # Install system dependencies
  sudo apt update
  sudo apt install ffmpeg libsdl2-dev libsdl2-mixer-dev

  # Install Python packages
  pip install -r requirements.txt

  # Run with Pi config
  python main.py config-pi.json

  Pi-specific notes:
  - The code auto-detects Pi via /proc/device-tree/model
  - Uses OpenGL ES 3.0 shaders automatically
  - Reduced glow radius and buffer size for better performance
  - config-pi.json has optimized settings for Pi

  FFmpeg requirement: pydub needs ffmpeg installed for MP3 decoding. On macOS: brew install ffmpeg. On Pi: sudo apt
  install ffmpeg.


