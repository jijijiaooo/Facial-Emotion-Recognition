# Raspberry Pi Kivy Troubleshooting Guide

## Problem: "Unnamed Window" Error

When running Kivy apps on Raspberry Pi, you may see:
```
[CRITICAL] [Window      ] Unable to find any valuable Window provider.
[CRITICAL] [App         ] Unable to get a Window, abort.
```

Or the window shows as "unnamed" in the title bar.

---

## Solutions Applied

### ✅ Fix 1: Set Window Title
Added proper title to the Kivy App class:
```python
class CompanionApp(App):
    title = "Emotion Companion"  # This fixes unnamed window
```

### ✅ Fix 2: Configure Kivy Before Imports
Added Raspberry Pi-specific configuration at the top of the file:
```python
# BEFORE any Kivy imports
os.environ['KIVY_WINDOW'] = 'sdl2'  # Use SDL2 backend
os.environ['KIVY_NO_CONSOLELOG'] = '1'  # Reduce spam

from kivy.config import Config
Config.set('kivy', 'window_icon', '')  # Prevent unnamed window
Config.set('graphics', 'maxfps', '30')  # Limit FPS for Pi
Config.set('graphics', 'multisamples', '0')  # Disable AA for speed
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')  # Fix touch
```

---

## Additional Raspberry Pi Setup

### 1. Install Required Dependencies

On Raspberry Pi OS:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade

# Install SDL2 dependencies (required for Kivy)
sudo apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    pkg-config \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    python3-setuptools \
    libgstreamer1.0-dev \
    git-core \
    gstreamer1.0-plugins-{bad,base,good,ugly} \
    gstreamer1.0-{omx,alsa} \
    python3-dev \
    libmtdev-dev \
    xclip \
    xsel

# Install Python dependencies
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install --upgrade Cython==0.29.33

# Install Kivy (this may take 30-60 minutes on Raspberry Pi!)
pip3 install kivy[base]==2.1.0
```

### 2. Enable OpenGL (if using Desktop Environment)

```bash
# Edit Raspberry Pi config
sudo raspi-config

# Navigate to: Advanced Options > GL Driver
# Select: G2 GL (Fake KMS) or G3 GL (Full KMS)
# Reboot
sudo reboot
```

### 3. Fix Display Issues

If you see "Unable to find any valuable Window provider":

```bash
# Create or edit Kivy config file
mkdir -p ~/.kivy
nano ~/.kivy/config.ini
```

Add this content:
```ini
[graphics]
window_state = visible
borderless = 0
fullscreen = 0
width = 480
height = 220
maxfps = 30
multisamples = 0

[input]
mouse = mouse,multitouch_on_demand

[kivy]
window_icon = 
log_level = info
```

---

## Running the App

### Method 1: Direct Run (Recommended)
```bash
cd /path/to/Facial-Emotion-Recognition-version-2.4
python3 apps/companions/kivy_companion_gui.py
```

### Method 2: Via Main Launcher
```bash
python3 run_companion.py
```

### Method 3: Headless Mode (SSH, no display)
If running via SSH without X11 forwarding:
```bash
# Use framebuffer
export DISPLAY=:0
python3 apps/companions/kivy_companion_gui.py
```

---

## Common Errors & Solutions

### Error 1: "Unable to find any valuable Window provider"

**Cause:** Missing SDL2 libraries

**Solution:**
```bash
sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
pip3 install --upgrade kivy[base]
```

### Error 2: "Could not initialize OpenGL"

**Cause:** OpenGL not enabled

**Solution:**
```bash
sudo raspi-config
# Advanced Options > GL Driver > G2 GL (Fake KMS)
sudo reboot
```

### Error 3: "Permission denied: /dev/video0"

**Cause:** No camera permissions

**Solution:**
```bash
sudo usermod -a -G video $USER
sudo reboot
```

### Error 4: App runs but FPS is <1

**Cause:** Too many models loaded (see main Raspberry Pi optimization guide)

**Solution:**
Edit `src/core/simple_emotion_detection.py` and use only 1 model:
```python
model_files = [
    'models/raf_db_simple_cnn.h5',  # Only this one!
]
```

### Error 5: "Unnamed window" title

**Cause:** Missing app title

**Solution:** Already fixed! The app now has `title = "Emotion Companion"`

### Error 6: Touch/Mouse not working

**Cause:** Multitouch configuration

**Solution:** Already fixed! Added:
```python
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
```

---

## Performance Tips for Raspberry Pi

### 1. Reduce Window Size
```python
Config.set('graphics', 'width', '400')  # Smaller
Config.set('graphics', 'height', '180')
```

### 2. Limit Frame Rate
```python
Config.set('graphics', 'maxfps', '20')  # Lower FPS = better performance
```

### 3. Disable Visual Effects
```python
Config.set('graphics', 'multisamples', '0')  # No antialiasing
```

### 4. Use Fewer Models
Only load 1 lightweight model (see MODEL_OMISSION_GUIDE.md)

---

## Verify Installation

Test if Kivy works:
```bash
python3 -c "import kivy; print(f'Kivy version: {kivy.__version__}')"
```

Expected output:
```
Kivy version: 2.1.0
```

Test simple Kivy app:
```python
python3 -c "
from kivy.app import App
from kivy.uix.label import Label

class TestApp(App):
    title = 'Test'
    def build(self):
        return Label(text='Kivy Works!')

if __name__ == '__main__':
    TestApp().run()
"
```

---

## Alternative: Use PyQt5 Instead

If Kivy continues to have issues, use the PyQt5 companion instead:

```bash
# Install PyQt5 (faster install than Kivy)
sudo apt-get install python3-pyqt5

# Run PyQt5 companion
python3 apps/companions/pyqt5_companion_gui.py
```

PyQt5 is often more reliable on Raspberry Pi!

---

## Debug Mode

Run with verbose output to see what's failing:

```bash
# Set environment variables for debugging
export KIVY_LOG_LEVEL=debug
export KIVY_NO_CONSOLELOG=0
python3 apps/companions/kivy_companion_gui.py
```

---

## System Requirements

### Minimum:
- Raspberry Pi 3B or newer
- Raspberry Pi OS (32-bit or 64-bit)
- 1GB RAM (recommend 2GB+)
- SDL2 libraries installed
- Python 3.7+

### Recommended:
- Raspberry Pi 4 (4GB+ RAM)
- Raspberry Pi OS 64-bit
- Desktop environment enabled
- OpenGL drivers enabled

---

## Quick Checklist

Before running the app, verify:

- [ ] SDL2 installed: `dpkg -l | grep libsdl2`
- [ ] Kivy installed: `python3 -c "import kivy"`
- [ ] Camera permissions: `ls -l /dev/video0`
- [ ] OpenGL enabled: `glxinfo | grep OpenGL` (if desktop)
- [ ] Display available: `echo $DISPLAY` (should show `:0`)
- [ ] App title set: Check line 1416 has `title = "Emotion Companion"`

---

## Summary of Changes Made

✅ **Fixed "unnamed window" issue:**
- Added `title = "Emotion Companion"` to CompanionApp class
- Added `Config.set('kivy', 'window_icon', '')` before imports

✅ **Optimized for Raspberry Pi:**
- Set `KIVY_WINDOW=sdl2` for better Pi support
- Limited FPS to 30 for performance
- Disabled antialiasing (multisamples)
- Fixed touch/mouse configuration

✅ **Better error handling:**
- Reduced console log spam
- Graceful fallback if detector fails

---

## Still Having Issues?

Try the PyQt5 companion instead - it's more stable on Raspberry Pi:
```bash
python3 apps/companions/pyqt5_companion_gui.py
```

Or run the basic emotion detection without GUI:
```bash
python3 src/core/simple_emotion_detection.py
```

---

Last Updated: October 11, 2025
