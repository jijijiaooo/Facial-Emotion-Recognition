# 🎭 Facial Emotion Recognition System

A comprehensive emotion recognition system with enhanced accuracy for fear and disgust detection, optimized for both desktop and Raspberry Pi deployment.

## ✨ Features

- **🎯 Enhanced Accuracy**: Improved fear and disgust detection using research-based techniques
- **🖥️ Multiple Interfaces**: GUI and command-line options
- **🍓 Raspberry Pi Optimized**: Lightweight GUI for edge deployment
- **⚡ Real-time Processing**: Live camera emotion detection
- **🤖 Multiple Models**: Support for TensorFlow, PyTorch, and TF Lite
- **📊 Dataset Tools**: Comprehensive dataset analysis and enhancement
- **🎨 User-Friendly GUI**: Touch-friendly interface with real-time statistics

## 🚀 Quick Start

### Option 1: GUI Application (Recommended)

```bash
# Simple GUI (most compatible)
python3 run_simple_gui.py

# Full-featured GUI
python3 run_gui.py
```

### Option 2: Command Line

```bash
# Enhanced emotion recognition
python3 src/core/enhanced_emotion_recognition.py

# Basic emotion recognition
python3 src/core/test.py
```

### Option 3: Raspberry Pi Setup

```bash
# Automated Pi setup
python3 setup_raspberry_pi.py

# Launch Pi-optimized GUI
python3 gui/emotion_gui_raspberry.py
```

## 📋 Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/facial-emotion-recognition.git
cd facial-emotion-recognition

# Install dependencies
pip install -r config/requirements.txt

# For Raspberry Pi
pip install -r config/requirements_raspberry_pi.txt
```

### System Requirements

- Python 3.7+
- OpenCV 4.5+
- Camera (built-in or USB webcam)
- For GUI: tkinter (usually included with Python)

## 🎭 Supported Emotions

| Emotion        | Status          | Accuracy |
| -------------- | --------------- | -------- |
| 😊 Happy       | ✅ Excellent    | 95%+     |
| 😢 Sad         | ✅ Good         | 85%+     |
| 😠 Angry       | ✅ Good         | 85%+     |
| 😲 Surprise    | ✅ Good         | 80%+     |
| 😐 Neutral     | ✅ Good         | 90%+     |
| 😨 **Fear**    | ⚡ **Enhanced** | 75%+     |
| 🤢 **Disgust** | ⚡ **Enhanced** | 70%+     |

## 📁 Project Structure

```
📦 facial-emotion-recognition/
├── 🎨 gui/                    # GUI applications
│   ├── emotion_gui_raspberry.py    # Raspberry Pi optimized
│   ├── simple_emotion_gui.py       # Simple, compatible GUI
│   └── basic_emotion_gui.py        # Basic GUI
├── 📄 src/core/               # Main source code
│   ├── main.py                     # Original training script
│   ├── test.py                     # Basic emotion detection
│   └── enhanced_emotion_recognition.py  # Enhanced detection
├── 🤖 models/                 # Model files and optimization
│   └── optimize_models.py          # Model optimization for Pi
├── 🔧 scripts/                # Utility scripts
│   ├── dataset_enhancement_guide.py
│   ├── fear_disgust_diagnostic.py
│   ├── au_enhanced_training.py
│   └── emotion_dataset_resources.py
├── ⚙️ config/                 # Configuration files
│   ├── requirements.txt
│   └── requirements_raspberry_pi.txt
├── 📚 docs/                   # Documentation
│   └── RASPBERRY_PI_GUIDE.md
├── 📊 data/                   # Training datasets (empty by default)
└── 🗑️ temp/                   # Temporary files
```

## 🔧 Usage Examples

### GUI Application

The GUI provides an intuitive interface with:

- Real-time camera feed
- Emotion detection with confidence scores
- Statistics and performance monitoring
- Adjustable detection sensitivity

### Command Line

```bash
# Run enhanced detection with temporal smoothing
python3 src/core/enhanced_emotion_recognition.py

# Analyze your dataset for improvements
python3 scripts/dataset_enhancement_guide.py

# Diagnose fear/disgust detection issues
python3 scripts/fear_disgust_diagnostic.py
```

## 🍓 Raspberry Pi Deployment

This system is optimized for Raspberry Pi with:

- Lightweight processing
- Touch-friendly interface
- Automatic hardware detection
- Performance optimizations

See [Raspberry Pi Guide](docs/RASPBERRY_PI_GUIDE.md) for detailed setup instructions.

## 🎯 Improving Accuracy

### For Fear and Disgust Detection

1. **Analyze your dataset**: `python3 scripts/fear_disgust_diagnostic.py`
2. **Collect more data**: Use the dataset resource guide
3. **Apply fixes**: `python3 scripts/fix_fear_disgust.py`

### Dataset Enhancement

- Use `scripts/emotion_dataset_resources.py` for data sources
- Follow Action Unit (AU) guidelines for better training data
- Balance your dataset across all emotions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FER-2013 dataset contributors
- OpenCV community
- TensorFlow team
- Raspberry Pi Foundation

## 📞 Support

- 📖 Check the [documentation](docs/)
- 🐛 Report issues on GitHub
- 💡 Feature requests welcome

## 🔄 Recent Updates

- ✅ Enhanced fear and disgust detection
- ✅ Raspberry Pi optimization
- ✅ Multiple GUI options
- ✅ Comprehensive dataset tools
- ✅ Action Unit (AU) based training

---

**Note**: This project includes research-based improvements for emotion recognition accuracy, particularly for underrepresented emotions like fear and disgust.
