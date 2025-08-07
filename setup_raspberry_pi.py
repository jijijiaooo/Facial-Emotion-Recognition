#!/usr/bin/env python3
"""
Raspberry Pi Setup Script for Emotion Recognition
Optimizes system for emotion recognition on Raspberry Pi OS
"""

import os
import subprocess
import sys
from pathlib import Path

class RaspberryPiSetup:
    def __init__(self):
        self.is_raspberry_pi = self.check_raspberry_pi()
        
    def check_raspberry_pi(self):
        """Check if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                return 'Raspberry Pi' in f.read()
        except:
            return False
    
    def install_system_dependencies(self):
        """Install system-level dependencies for Raspberry Pi"""
        print("📦 Installing system dependencies...")
        
        if not self.is_raspberry_pi:
            print("⚠️ Not running on Raspberry Pi - skipping system setup")
            return True
        
        # System packages needed for OpenCV and camera
        packages = [
            'python3-opencv',
            'python3-pil',
            'python3-pil.imagetk',
            'python3-numpy',
            'libatlas-base-dev',
            'libjasper-dev',
            'libqtgui4',
            'libqt4-test',
            'libhdf5-dev',
            'libhdf5-serial-dev',
            'libatlas-base-dev',
            'libjasper-dev',
            'libqtgui4',
            'libqt4-test'
        ]
        
        try:
            # Update package list
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            
            # Install packages
            cmd = ['sudo', 'apt', 'install', '-y'] + packages
            subprocess.run(cmd, check=True)
            
            print("✅ System dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install system dependencies: {e}")
            return False
    
    def setup_camera(self):
        """Setup camera for Raspberry Pi"""
        print("📷 Setting up camera...")
        
        if not self.is_raspberry_pi:
            print("⚠️ Not on Raspberry Pi - skipping camera setup")
            return True
        
        try:
            # Enable camera interface
            print("  Enabling camera interface...")
            subprocess.run(['sudo', 'raspi-config', 'nonint', 'do_camera', '0'], check=True)
            
            # Add user to video group
            username = os.getenv('USER')
            if username:
                subprocess.run(['sudo', 'usermod', '-a', '-G', 'video', username], check=True)
                print(f"  Added {username} to video group")
            
            print("✅ Camera setup complete")
            print("💡 You may need to reboot for camera changes to take effect")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Camera setup warning: {e}")
            return False
    
    def optimize_memory(self):
        """Optimize memory settings for Raspberry Pi"""
        print("🧠 Optimizing memory settings...")
        
        if not self.is_raspberry_pi:
            return True
        
        try:
            # Increase GPU memory split for camera
            config_file = '/boot/config.txt'
            
            # Check if we can modify config
            if os.path.exists(config_file):
                print("  Setting GPU memory split to 128MB...")
                
                # Read current config
                with open(config_file, 'r') as f:
                    config_content = f.read()
                
                # Add or modify gpu_mem setting
                if 'gpu_mem=' in config_content:
                    # Replace existing setting
                    lines = config_content.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('gpu_mem='):
                            lines[i] = 'gpu_mem=128'
                            break
                    config_content = '\n'.join(lines)
                else:
                    # Add new setting
                    config_content += '\n# GPU memory for camera\ngpu_mem=128\n'
                
                # Write back (requires sudo)
                temp_file = '/tmp/config.txt'
                with open(temp_file, 'w') as f:
                    f.write(config_content)
                
                subprocess.run(['sudo', 'cp', temp_file, config_file], check=True)
                os.remove(temp_file)
                
                print("✅ Memory optimization complete")
                print("💡 Reboot required for memory changes")
                return True
            
        except Exception as e:
            print(f"⚠️ Memory optimization failed: {e}")
            return False
    
    def install_python_dependencies(self):
        """Install Python dependencies optimized for Pi"""
        print("🐍 Installing Python dependencies...")
        
        # Lightweight requirements for Raspberry Pi
        pi_requirements = [
            'opencv-python-headless',  # Headless version for Pi
            'Pillow',
            'numpy',
            'matplotlib',  # For any plotting needs
        ]
        
        # Optional ML libraries
        optional_requirements = [
            'tensorflow-lite-runtime',  # Lightweight TF for Pi
            'scikit-learn',
        ]
        
        try:
            # Install core requirements
            for package in pi_requirements:
                print(f"  Installing {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
            
            # Try to install optional packages
            for package in optional_requirements:
                try:
                    print(f"  Installing {package} (optional)...")
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    print(f"  ⚠️ Could not install {package} - skipping")
            
            print("✅ Python dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Python dependencies: {e}")
            return False
    
    def create_desktop_shortcut(self):
        """Create desktop shortcut for the GUI"""
        print("🖥️ Creating desktop shortcut...")
        
        try:
            desktop_dir = Path.home() / 'Desktop'
            if not desktop_dir.exists():
                desktop_dir = Path.home()  # Fallback to home directory
            
            script_dir = Path(__file__).parent.absolute()
            
            shortcut_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Emotion Recognition
Comment=Facial Emotion Recognition System
Exec=python3 {script_dir}/run_gui.py
Icon={script_dir}/docs/icon.png
Terminal=false
Categories=Application;AudioVideo;
"""
            
            shortcut_path = desktop_dir / 'EmotionRecognition.desktop'
            
            with open(shortcut_path, 'w') as f:
                f.write(shortcut_content)
            
            # Make executable
            os.chmod(shortcut_path, 0o755)
            
            print(f"✅ Desktop shortcut created: {shortcut_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not create desktop shortcut: {e}")
            return False
    
    def create_startup_script(self):
        """Create startup script for auto-launch"""
        print("🚀 Creating startup script...")
        
        try:
            script_dir = Path(__file__).parent.absolute()
            
            startup_script = f"""#!/bin/bash
# Emotion Recognition Startup Script

# Wait for desktop to load
sleep 10

# Launch emotion recognition GUI
cd {script_dir}
python3 run_gui.py

# Log any errors
echo "Emotion Recognition started at $(date)" >> /tmp/emotion_recognition.log
"""
            
            startup_path = script_dir / 'start_emotion_recognition.sh'
            
            with open(startup_path, 'w') as f:
                f.write(startup_script)
            
            os.chmod(startup_path, 0o755)
            
            print(f"✅ Startup script created: {startup_path}")
            print("💡 To auto-start on boot, add this to your autostart:")
            print(f"   {startup_path}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Could not create startup script: {e}")
            return False
    
    def test_installation(self):
        """Test the installation"""
        print("🧪 Testing installation...")
        
        tests_passed = 0
        total_tests = 4
        
        # Test 1: Import OpenCV
        try:
            import cv2
            print("  ✅ OpenCV import successful")
            tests_passed += 1
        except ImportError:
            print("  ❌ OpenCV import failed")
        
        # Test 2: Import PIL
        try:
            from PIL import Image, ImageTk
            print("  ✅ PIL import successful")
            tests_passed += 1
        except ImportError:
            print("  ❌ PIL import failed")
        
        # Test 3: Camera access
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.release()
                print("  ✅ Camera access successful")
                tests_passed += 1
            else:
                print("  ❌ Camera access failed")
        except:
            print("  ❌ Camera test failed")
        
        # Test 4: GUI libraries
        try:
            import tkinter as tk
            print("  ✅ Tkinter available")
            tests_passed += 1
        except ImportError:
            print("  ❌ Tkinter not available")
        
        print(f"\n📊 Test Results: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! System ready for emotion recognition.")
            return True
        else:
            print("⚠️ Some tests failed. Check the errors above.")
            return False
    
    def run_setup(self):
        """Run complete setup process"""
        print("🍓 RASPBERRY PI EMOTION RECOGNITION SETUP")
        print("=" * 50)
        
        if self.is_raspberry_pi:
            print("✅ Running on Raspberry Pi")
        else:
            print("⚠️ Not detected as Raspberry Pi - running basic setup")
        
        setup_steps = [
            ("Installing system dependencies", self.install_system_dependencies),
            ("Setting up camera", self.setup_camera),
            ("Optimizing memory", self.optimize_memory),
            ("Installing Python dependencies", self.install_python_dependencies),
            ("Creating desktop shortcut", self.create_desktop_shortcut),
            ("Creating startup script", self.create_startup_script),
            ("Testing installation", self.test_installation),
        ]
        
        successful_steps = 0
        
        for step_name, step_function in setup_steps:
            print(f"\n{step_name}...")
            try:
                if step_function():
                    successful_steps += 1
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
        
        print(f"\n📊 SETUP COMPLETE")
        print("=" * 20)
        print(f"Successful steps: {successful_steps}/{len(setup_steps)}")
        
        if successful_steps >= len(setup_steps) - 1:  # Allow 1 failure
            print("\n🎉 Setup completed successfully!")
            print("\n🚀 To start the emotion recognition GUI:")
            print("   python3 run_gui.py")
            print("\n💡 Or use the desktop shortcut if created")
            
            if self.is_raspberry_pi:
                print("\n🔄 Recommended: Reboot your Raspberry Pi to apply all changes")
                print("   sudo reboot")
        else:
            print("\n⚠️ Setup completed with issues. Check the errors above.")
        
        return successful_steps >= len(setup_steps) - 1

def main():
    setup = RaspberryPiSetup()
    return setup.run_setup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)