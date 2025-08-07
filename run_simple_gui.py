#!/usr/bin/env python3
"""
Direct launcher for Simple Emotion Recognition GUI
Bypasses compatibility checks for quick testing
"""

import sys
import os
from pathlib import Path

# Add paths
script_dir = Path(__file__).parent.absolute()
gui_dir = script_dir / "gui"
sys.path.insert(0, str(gui_dir))

def main():
    print("🎭 Simple Emotion Recognition GUI")
    print("=" * 35)
    
    try:
        # Import and run the simple GUI directly
        import simple_emotion_gui
        simple_emotion_gui.main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have the required dependencies:")
        print("   pip install opencv-python pillow numpy")
        if 'tensorflow' in str(e).lower():
            print("   pip install tensorflow  # Optional for better accuracy")
        return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())