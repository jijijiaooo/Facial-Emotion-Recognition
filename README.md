# Interactive Emotion Recognition Companion

A comprehensive facial emotion recognition system with interactive companion applications, similar to virtual pet games like Pou, but with advanced emotion detection capabilities.

## ✨ Features

- 🎭 **Interactive Virtual Companion** - Like Pou, but responds to your real emotions
- 📹 **Real-time Emotion Detection** - Uses advanced Action Units (AU) analysis
- 💬 **Intelligent Chat System** - Contextual conversations based on your mood
- 🎮 **Fun Activities** - Games, music, meditation, and more
- 📊 **Emotion Analytics** - Track your emotional patterns over time
- 🎨 **Multiple Interfaces** - Modern PyQt5 GUI and Terminal options

## 🚀 Quick Start

### Option 1: Run Interactive Companion (Recommended)

```bash
# Activate virtual environment
source venv/bin/activate

# Run the companion (includes emotion detection)
python run_companion.py
```

### Option 2: Run Emotion Detection Only

```bash
# Run enhanced emotion detection
python run_detection.py
```

### Option 3: Advanced Launcher

```bash
# Run the advanced launcher with multiple options
python run_latest_app.py

# Terminal Companion (No GUI Required)
python apps/companions/robust_companion_app.py
```

## 📁 Project Structure

```
📦 Interactive Emotion Recognition Companion
├── 📁 apps/                          # Main applications
│   ├── 📁 companions/                # Interactive companion apps
│   │   ├── pyqt5_companion_gui.py   # PyQt5 GUI companion (recommended)
│   │   ├── tkinter_companion_gui.py # Tkinter GUI companion
│   │   ├── robust_companion_app.py  # Terminal companion
│   │   └── simple_pixel_companion.py # Pixel art companion
│   └── 📁 utils/                     # Utility applications
├── 📁 src/                           # Core source code
│   └── 📁 core/                      # Core emotion detection
│       └── simple_emotion_detection.py # Enhanced emotion detection with AUs
├── 📁 gui/                           # Additional GUI components
├── 📁 models/                        # Pre-trained models
├── 📁 config/                        # Configuration files
├── 📁 training/                      # Model training scripts
├── 📁 scripts/                       # Utility scripts
└── run_latest_app.py                 # Main launcher
```

## 🎮 Companion Features

### Interactive Companion (Like Pou!)

- **Emotional Responses** - Companion reacts to your detected emotions
- **Happiness & Energy Bars** - Visual stats that change based on interactions
- **Mood System** - Companion has different moods (happy, sad, excited, calm, worried)
- **Personalization** - Change companion name and settings

### Activities

- 🎲 **Random Mood Boost** - Cheer up your companion
- 💝 **Give Virtual Gifts** - Flowers, cake, presents, and more
- 🎵 **Play Music Together** - Dance and listen to music
- 🎨 **Change Appearance** - Modify companion colors and style
- 🏃 **Exercise** - Do workouts together
- 🧘 **Meditate** - Find inner peace together
- 🎪 **Story Time** - Listen to stories
- 🎯 **Play Games** - Riddles, word games, and more

### Communication

- 💬 **Real-time Chat** - Intelligent conversations
- 📹 **Camera Emotion Detection** - Sees your emotions through camera
- 😊 **Manual Emotion Input** - Tell companion how you feel
- 🔔 **Periodic Check-ins** - Companion asks how you're doing

## 🔧 Technical Features

### Enhanced Emotion Detection

- **Action Units (AUs)** - Facial muscle movement analysis
- **7 Emotions** - Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- **High Accuracy** - Improved fear and disgust detection
- **Real-time Processing** - Live camera feed analysis

### Supported Platforms

- **macOS** - Full support with PyQt5
- **Windows** - Full support
- **Linux** - Full support
- **Raspberry Pi** - Optimized lightweight version

## 📋 Requirements

### System Requirements

- Python 3.9+
- Webcam (for emotion detection)
- 4GB RAM minimum
- Modern CPU (GPU optional but recommended)

### Dependencies

```bash
# Core dependencies
pip install opencv-python numpy Pillow

# GUI dependencies
pip install PyQt5  # For PyQt5 GUI (recommended)
# tkinter is usually included with Python

# ML dependencies (optional, for training)
pip install tensorflow keras torch torchvision
```

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd "Interactive Emotion Recognition Companion"
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r config/requirements.txt
```

4. **Run the application**

```bash
python run_latest_app.py
```

## 🎯 Usage Guide

### First Time Setup

1. Run the launcher: `python run_latest_app.py`
2. Choose option 1 (PyQt5 GUI Companion)
3. Allow camera access when prompted
4. Start interacting with your companion!

### Companion Interaction

- **Camera Detection**: Click "Start Camera" to enable real-time emotion detection
- **Manual Input**: Use emotion buttons to tell companion how you feel
- **Chat**: Switch to Chat tab for conversations
- **Activities**: Try different activities in the Activities tab
- **Stats**: View your interaction statistics
- **Settings**: Customize companion name and behavior

### Tips for Best Experience

- Ensure good lighting for camera detection
- Position your face clearly in the camera view
- Try different activities to see how companion responds
- Chat regularly to build relationship with companion

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Emotion recognition research community
- OpenCV and PyQt5 developers
- Virtual pet game inspiration (Pou, Tamagotchi)
- Facial Action Coding System (FACS) research

## 📞 Support

If you encounter any issues:

1. Check the troubleshooting section in docs/
2. Run the compatibility checker: `python apps/utils/check_gui_compatibility.py`
3. Open an issue on GitHub

---

**Enjoy your interactive emotion companion! 🎭✨**
