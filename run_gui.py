#!/usr/bin/env python3
"""
Launcher script for Emotion Recognition GUI
Handles path setup and launches the GUI application
"""

import os
import sys
from pathlib import Path

def setup_paths():
    """Setup Python paths for imports"""
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Add necessary paths
    paths_to_add = [
        script_dir,  # Root directory
        script_dir / "src" / "core",  # Core modules
        script_dir / "gui",  # GUI modules
    ]
    
    for path in paths_to_add:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    # Check core dependencies
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import PIL
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    # Optional dependencies
    optional_missing = []
    try:
        import tensorflow
    except ImportError:
        optional_missing.append("tensorflow")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install with:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    if optional_missing:
        print("⚠️ Optional dependencies missing:")
        for dep in optional_missing:
            print(f"   - {dep}")
        print("   (Basic emotion detection will be used)")
    
    return True

def check_model_files():
    """Check if model files are available"""
    script_dir = Path(__file__).parent.absolute()
    model_dir = script_dir / "models"
    
    model_files = [
        "model_file_30epochs.h5",
        "emotion_model.tflite",
        "emotion_model_quantized.tflite"
    ]
    
    found_models = []
    for model_file in model_files:
        if (model_dir / model_file).exists():
            found_models.append(model_file)
    
    if found_models:
        print(f"✅ Found models: {', '.join(found_models)}")
    else:
        print("⚠️ No model files found - using basic detection")
    
    return len(found_models) > 0

def check_camera():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            print("✅ Camera available")
            return True
        else:
            print("⚠️ Camera not available")
            return False
    except Exception as e:
        print(f"⚠️ Camera check failed: {e}")
        return False

def main():
    print("🎭 Emotion Recognition GUI Launcher")
    print("=" * 40)
    
    # Setup paths
    setup_paths()
    
    # Check system requirements
    print("\n🔍 Checking system requirements...")
    
    if not check_dependencies():
        print("\n❌ Cannot start GUI - missing dependencies")
        return 1
    
    check_model_files()
    check_camera()
    
    # Launch GUI
    print("\n🚀 Starting GUI application...")
    
    try:
        # Import GUI module
        import sys
        import os
        
        # Add GUI directory to path
        gui_dir = os.path.join(os.path.dirname(__file__), 'gui')
        if gui_dir not in sys.path:
            sys.path.insert(0, gui_dir)
        
        # Try simple GUI first (more compatible)
        try:
            import simple_emotion_gui
            simple_emotion_gui.main()
        except Exception as simple_error:
            print(f"⚠️ Simple GUI failed: {simple_error}")
            print("💡 Trying Raspberry Pi GUI...")
            import emotion_gui_raspberry
            emotion_gui_raspberry.main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Trying alternative import method...")
        
        # Alternative: run as subprocess to avoid import issues
        try:
            import subprocess
            gui_script = os.path.join(os.path.dirname(__file__), 'gui', 'emotion_gui_raspberry.py')
            subprocess.run([sys.executable, gui_script])
            return 0
        except Exception as e2:
            print(f"❌ Alternative method failed: {e2}")
            return 1
    
    except Exception as e:
        print(f"❌ GUI error: {e}")
        print("💡 This might be a compatibility issue with your system")
        
        # Offer command-line alternative
        print("\n🔧 Would you like to try the command-line version instead?")
        try:
            response = input("Type 'y' for yes, 'n' for no: ").lower()
            if response == 'y':
                print("🚀 Starting command-line emotion recognition...")
                import subprocess
                cmd_script = os.path.join(os.path.dirname(__file__), 'src', 'core', 'enhanced_emotion_recognition.py')
                if os.path.exists(cmd_script):
                    subprocess.run([sys.executable, cmd_script])
                else:
                    print("❌ Command-line version not found")
                return 0
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
        except Exception:
            pass
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())