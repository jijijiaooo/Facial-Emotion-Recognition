#!/usr/bin/env python3
"""
Installation script for Interactive Emotion Recognition Companion
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True

def main():
    print("🎭 Interactive Emotion Recognition Companion - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if virtual environment exists
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        if not run_command(f"{sys.executable} -m venv venv", "Virtual environment creation"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install core dependencies
    core_deps = [
        "opencv-python>=4.5.0",
        "numpy>=1.21.0", 
        "Pillow>=8.0.0",
        "PyQt5>=5.15.0"
    ]
    
    print("📦 Installing core dependencies...")
    for dep in core_deps:
        if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
            print(f"⚠️ Failed to install {dep}, continuing...")
    
    # Optional: Install ML dependencies
    print("\n🤖 Would you like to install machine learning dependencies?")
    print("   (Required for advanced emotion detection features)")
    choice = input("   Install ML dependencies? (y/N): ").lower().strip()
    
    if choice in ['y', 'yes']:
        ml_deps = [
            "tensorflow>=2.10.0",
            "keras>=2.10.0"
        ]
        
        for dep in ml_deps:
            if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
                print(f"⚠️ Failed to install {dep}, continuing...")
    
    print("\n🎉 Installation completed!")
    print("\n🚀 To run the application:")
    print(f"   1. Activate virtual environment: {activate_cmd}")
    print("   2. Run the launcher: python run_latest_app.py")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()