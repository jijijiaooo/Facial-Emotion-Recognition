#!/usr/bin/env python3
"""
Kivy Diagnostic Tool for Raspberry Pi
Checks Kivy installation and window configuration
"""

import sys
import os

print("=" * 60)
print("KIVY DIAGNOSTIC TOOL FOR RASPBERRY PI")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check if running on Raspberry Pi
print("\n2. Raspberry Pi Detection:")
try:
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip('\x00')
            print(f"   ✅ Raspberry Pi Detected: {model}")
    else:
        print("   ⚠️  Not running on Raspberry Pi (or /proc/device-tree/model not found)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Check RAM
print("\n3. RAM Available:")
try:
    with open('/proc/meminfo', 'r') as f:
        meminfo = f.read()
        total_kb = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
        available_kb = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1])
        print(f"   Total RAM: {total_kb / 1024 / 1024:.1f} GB")
        print(f"   Available RAM: {available_kb / 1024 / 1024:.1f} GB")
except Exception as e:
    print(f"   ⚠️  Could not read memory info: {e}")

# Check Kivy installation
print("\n4. Kivy Installation:")
try:
    import kivy
    print(f"   ✅ Kivy installed: v{kivy.__version__}")
except ImportError:
    print("   ❌ Kivy NOT installed")
    print("   Install with: pip3 install kivy")
    sys.exit(1)

# Check Kivy dependencies
print("\n5. Kivy Dependencies:")
dependencies = {
    'SDL2': 'sdl2',
    'PIL/Pillow': 'PIL',
    'NumPy': 'numpy'
}

for name, module in dependencies.items():
    try:
        __import__(module)
        print(f"   ✅ {name} available")
    except ImportError:
        print(f"   ⚠️  {name} NOT available")

# Check SDL2 libraries
print("\n6. SDL2 Libraries:")
sdl2_libs = [
    'libSDL2-2.0.so.0',
    'libSDL2_image-2.0.so.0',
    'libSDL2_mixer-2.0.so.0',
    'libSDL2_ttf-2.0.so.0'
]

for lib in sdl2_libs:
    try:
        import ctypes
        ctypes.CDLL(lib)
        print(f"   ✅ {lib} found")
    except Exception:
        print(f"   ⚠️  {lib} NOT found")

# Check environment variables
print("\n7. Environment Variables:")
env_vars = {
    'KIVY_WINDOW': 'Window backend',
    'KIVY_GL_BACKEND': 'GL backend',
    'SDL_VIDEODRIVER': 'Video driver',
    'DISPLAY': 'Display'
}

for var, desc in env_vars.items():
    value = os.environ.get(var, 'Not set')
    print(f"   {var} ({desc}): {value}")

# Test basic Kivy window creation
print("\n8. Testing Kivy Window Creation:")
try:
    # Set environment variables
    os.environ['KIVY_NO_CONSOLELOG'] = '1'
    os.environ['KIVY_WINDOW'] = 'sdl2'
    
    from kivy.config import Config
    Config.set('kivy', 'window_icon', '')
    Config.set('graphics', 'window_state', 'visible')
    Config.set('graphics', 'width', '400')
    Config.set('graphics', 'height', '300')
    Config.write()
    
    from kivy.app import App
    from kivy.uix.label import Label
    from kivy.core.window import Window
    
    class TestApp(App):
        title = "Kivy Test Window"
        icon = ''
        
        def build(self):
            Window.set_title("Kivy Test Window")
            return Label(text='✅ Kivy Window Created Successfully!\nClose this window to continue.')
    
    print("   Opening test window...")
    print("   If you see a window titled 'Kivy Test Window', Kivy is working!")
    print("   Close the window to continue.\n")
    
    TestApp().run()
    
    print("\n   ✅ Kivy window test PASSED!")
    
except Exception as e:
    print(f"\n   ❌ Kivy window test FAILED: {e}")
    print("\n   Troubleshooting steps:")
    print("   1. Install SDL2: sudo apt-get install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev")
    print("   2. Reinstall Kivy: pip3 install --upgrade --force-reinstall kivy")
    print("   3. Check display: echo $DISPLAY (should show :0 or :0.0)")
    print("   4. Try different backend: export KIVY_WINDOW=x11")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
