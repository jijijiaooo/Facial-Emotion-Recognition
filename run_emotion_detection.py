#!/usr/bin/env python3
"""
Smart Emotion Detection Launcher
Automatically selects the best working version for your system
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check what dependencies are available"""
    available = {}
    
    # Check OpenCV
    try:
        import cv2
        available['opencv'] = cv2.__version__
        print(f"✅ OpenCV {cv2.__version__} available")
    except ImportError:
        available['opencv'] = None
        print("❌ OpenCV not available")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        available['tensorflow'] = tf.__version__
        print(f"✅ TensorFlow {tf.__version__} available")
    except ImportError:
        available['tensorflow'] = None
        print("⚠️ TensorFlow not available (basic detection only)")
    
    # Check PIL
    try:
        import PIL
        available['pil'] = PIL.__version__
        print(f"✅ PIL {PIL.__version__} available")
    except ImportError:
        available['pil'] = None
        print("⚠️ PIL not available (GUI may not work)")
    
    # Check tkinter
    try:
        import tkinter
        available['tkinter'] = True
        print("✅ Tkinter available")
    except ImportError:
        available['tkinter'] = False
        print("⚠️ Tkinter not available (no GUI)")
    
    return available

def check_model_files():
    """Check what model files are available"""
    model_paths = [
        'model_file_30epochs.h5',
        'models/model_file_30epochs.h5',
        'src/core/model_file_30epochs.h5'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ Model found: {path} ({size_mb:.1f} MB)")
            return path
    
    print("⚠️ No model files found - will use basic detection")
    return None

def check_cascade_files():
    """Check if Haar cascade files are available"""
    try:
        import cv2
        # Try built-in cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            print("✅ Built-in Haar cascade available")
            return True
    except:
        pass
    
    # Try local files
    cascade_paths = [
        'haarcascade_frontalface_default.xml',
        'config/haarcascade_frontalface_default.xml'
    ]
    
    for path in cascade_paths:
        if os.path.exists(path):
            print(f"✅ Local Haar cascade found: {path}")
            return True
    
    print("⚠️ No Haar cascade files found - will use fallback detection")
    return False

def run_simple_detection():
    """Run the simple, most compatible version"""
    print("\n🚀 Starting Simple Emotion Detection...")
    try:
        script_path = "src/core/simple_emotion_detection.py"
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path])
        else:
            print(f"❌ Script not found: {script_path}")
            return False
        return True
    except Exception as e:
        print(f"❌ Simple detection failed: {e}")
        return False

def run_enhanced_detection():
    """Run the enhanced version"""
    print("\n🚀 Starting Enhanced Emotion Detection...")
    try:
        script_path = "src/core/enhanced_emotion_recognition.py"
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path])
        else:
            print(f"❌ Script not found: {script_path}")
            return False
        return True
    except Exception as e:
        print(f"❌ Enhanced detection failed: {e}")
        return False

def run_basic_detection():
    """Run the basic test version"""
    print("\n🚀 Starting Basic Emotion Detection...")
    try:
        script_path = "src/core/test.py"
        if os.path.exists(script_path):
            subprocess.run([sys.executable, script_path])
        else:
            print(f"❌ Script not found: {script_path}")
            return False
        return True
    except Exception as e:
        print(f"❌ Basic detection failed: {e}")
        return False

def run_gui():
    """Try to run GUI version"""
    print("\n🚀 Attempting to start GUI...")
    
    gui_scripts = [
        ("Simple GUI", "run_simple_gui.py"),
        ("Full GUI", "run_gui.py"),
        ("Direct Simple GUI", "gui/simple_emotion_gui.py")
    ]
    
    for name, script in gui_scripts:
        if os.path.exists(script):
            try:
                print(f"Trying {name}...")
                subprocess.run([sys.executable, script], timeout=5)
                return True
            except subprocess.TimeoutExpired:
                print(f"⚠️ {name} started but may have GUI issues")
                return True
            except Exception as e:
                print(f"⚠️ {name} failed: {e}")
                continue
    
    print("❌ All GUI attempts failed")
    return False

def main():
    print("🎭 SMART EMOTION DETECTION LAUNCHER")
    print("=" * 50)
    print("Automatically selecting the best version for your system...\n")
    
    # Check system capabilities
    print("🔍 Checking system capabilities...")
    deps = check_dependencies()
    model_available = check_model_files() is not None
    cascade_available = check_cascade_files()
    
    print(f"\n📊 System Summary:")
    print(f"   OpenCV: {'✅' if deps['opencv'] else '❌'}")
    print(f"   TensorFlow: {'✅' if deps['tensorflow'] else '❌'}")
    print(f"   GUI Support: {'✅' if deps['tkinter'] and deps['pil'] else '❌'}")
    print(f"   Model Available: {'✅' if model_available else '❌'}")
    print(f"   Face Detection: {'✅' if cascade_available else '⚠️ Fallback'}")
    
    # Determine best option
    if not deps['opencv']:
        print("\n❌ OpenCV not available - cannot run emotion detection")
        print("💡 Install with: pip install opencv-python")
        return 1
    
    print(f"\n🎯 Recommended Options:")
    
    options = []
    
    # GUI options (if available)
    if deps['tkinter'] and deps['pil']:
        options.append(("1", "GUI Application (Recommended)", run_gui))
    
    # Command line options
    if model_available and cascade_available:
        options.append((str(len(options)+1), "Enhanced Detection (Full Features)", run_enhanced_detection))
    
    options.append((str(len(options)+1), "Simple Detection (Most Compatible)", run_simple_detection))
    
    if model_available:
        options.append((str(len(options)+1), "Basic Detection (Original)", run_basic_detection))
    
    # Show options
    for num, desc, _ in options:
        print(f"   {num}. {desc}")
    
    print(f"   q. Quit")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\n🚀 Select option (1-{len(options)} or q): ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                return 0
            
            # Find matching option
            for num, desc, func in options:
                if choice == num:
                    print(f"\n▶️ Starting: {desc}")
                    success = func()
                    if success:
                        return 0
                    else:
                        print(f"\n⚠️ {desc} failed. Try another option.")
                        break
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

if __name__ == "__main__":
    exit(main())