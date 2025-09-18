#!/usr/bin/env python3
"""
Direct Companion Launcher - Runs the best companion app directly
"""

import os
import sys
import subprocess

def main():
    print("INTERACTIVE EMOTION COMPANION")
    print("=" * 50)
    print("Starting the best companion experience...")
    print()
    
    # Primary app: Kivy GUI Companion (includes enhanced emotion detection)
    primary_app = "apps/companions/kivy_companion_gui.py"
    
    # Fallback apps in order of preference
    fallback_apps = [
        ("PyQt5 GUI Companion", "apps/companions/pyqt5_companion_gui.py"),
        ("Robust Terminal Companion", "apps/companions/robust_companion_app.py"),
        ("Enhanced Emotion Detection", "src/core/simple_emotion_detection.py")
    ]
    
    # Try to run the primary app
    if os.path.exists(primary_app):
        print("Starting Kivy GUI Companion (Best Experience)")
        print("Includes: Interactive companion + Enhanced emotion detection + Retro pixelated interface")
        print(f"Running: {primary_app}")
        print("=" * 50)
        
        try:
            subprocess.run([sys.executable, primary_app])
            return 0
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return 0
        except Exception as e:
            print(f"Error running Kivy companion: {e}")
            print("Trying fallback options...")
    else:
        print("Kivy GUI Companion not found, trying alternatives...")
    
    # Try fallback apps
    for name, path in fallback_apps:
        if os.path.exists(path):
            print(f"\nStarting: {name}")
            print(f"Running: {path}")
            print("=" * 50)
            
            try:
                subprocess.run([sys.executable, path])
                return 0
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return 0
            except Exception as e:
                print(f"Error running {name}: {e}")
                continue
    
    # If we get here, nothing worked
    print("No applications could be started!")
    print("Try running: python install.py")
    return 1

if __name__ == "__main__":
    exit(main())