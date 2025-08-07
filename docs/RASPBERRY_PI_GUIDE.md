# 🍓 Raspberry Pi Emotion Recognition Guide

Complete guide for setting up and running the Emotion Recognition system on Raspberry Pi OS.

## 🚀 Quick Setup

### 1. Automated Setup (Recommended)

```bash
# Run the automated setup script
python3 setup_raspberry_pi.py

# Reboot after setup
sudo reboot

# Launch the GUI
python3 run_gui.py
```

### 2. Manual Setup

#### System Requirements

- Raspberry Pi 3B+ or newer (Pi 4 recommended)
- Raspberry Pi OS (32-bit or 64-bit)
- Camera module or USB webcam
- At least 2GB RAM (4GB+ recommended)
- 8GB+ SD card with fast read/write speeds

#### Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3-opencv python3-pil python3-pil.imagetk python3-numpy libatlas-base-dev
```

#### Enable Camera

```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
```

#### Install Python Dependencies

```bash
pip3 install -r config/requirements_raspberry_pi.txt
```

## 🎭 Running the Application

### GUI Application

```bash
# Launch the graphical interface
python3 run_gui.py
```

### Command Line Version

```bash
# Run enhanced emotion recognition
python3 src/core/enhanced_emotion_recognition.py
```

## ⚙️ Optimization for Raspberry Pi

### Performance Settings

#### 1. GPU Memory Split

Add to `/boot/config.txt`:

```
gpu_mem=128
```

#### 2. Camera Settings

The GUI automatically optimizes camera settings for Pi:

- Resolution: 640x480 (lower for better performance)
- FPS: 15 (reduced from 30)
- Buffer size: 1 (minimal lag)

#### 3. Model Selection

The system automatically chooses the best model:

1. **TensorFlow Lite** (fastest, if available)
2. **Quantized TF Lite** (smallest, good for Pi Zero)
3. **Original H5 model** (most accurate, slower)
4. **Basic detection** (fallback, no ML required)

### Memory Management

- Uses lightweight OpenCV (headless version)
- Optimized image processing
- Reduced frame buffer
- Efficient face detection

## 🎯 Features Optimized for Pi

### GUI Features

- **Raspberry Pi Detection**: Automatically detects Pi hardware
- **Performance Monitoring**: Real-time FPS and resource usage
- **Adaptive Quality**: Adjusts processing based on performance
- **Touch-Friendly**: Large buttons for touchscreen displays
- **Low Resource Usage**: Minimal CPU and memory footprint

### Emotion Detection

- **7 Emotions**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Real-time Processing**: Optimized for Pi's ARM processor
- **Confidence Thresholding**: Adjustable sensitivity
- **Statistics Tracking**: Emotion frequency analysis

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Working

```bash
# Check camera connection
vcgencmd get_camera

# Should show: supported=1 detected=1

# Test camera
raspistill -o test.jpg
```

#### Low Performance

1. **Reduce resolution**: Edit camera settings in GUI
2. **Lower FPS**: Increase frame skip in code
3. **Use TF Lite**: Ensure optimized models are available
4. **Close other apps**: Free up system resources

#### Import Errors

```bash
# Reinstall OpenCV
sudo apt install python3-opencv

# Or use pip version
pip3 install opencv-python-headless
```

#### GUI Issues

```bash
# Install tkinter if missing
sudo apt install python3-tk

# Check display
echo $DISPLAY
```

### Performance Tips

#### For Pi 3B+

- Use 480x360 resolution
- Enable only essential emotions
- Use quantized models
- Limit to 10 FPS

#### For Pi 4

- Use 640x480 resolution
- All emotions enabled
- Use TF Lite models
- Up to 20 FPS possible

#### For Pi Zero

- Use 320x240 resolution
- Basic detection only
- 5 FPS maximum
- Consider headless operation

## 📱 Touchscreen Setup

### Official Pi Touchscreen

The GUI is optimized for the official 7" touchscreen:

- Large, touch-friendly buttons
- Clear emotion display
- Swipe gestures (future feature)

### Setup Commands

```bash
# Rotate display if needed
echo 'lcd_rotate=2' | sudo tee -a /boot/config.txt

# Calibrate touch (if needed)
sudo apt install xinput-calibrator
```

## 🚀 Auto-Start on Boot

### Method 1: Desktop Autostart

```bash
# Create autostart directory
mkdir -p ~/.config/autostart

# Create desktop entry
cat > ~/.config/autostart/emotion-recognition.desktop << EOF
[Desktop Entry]
Type=Application
Name=Emotion Recognition
Exec=python3 /home/pi/emotion-recognition/run_gui.py
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
EOF
```

### Method 2: Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/emotion-recognition.service << EOF
[Unit]
Description=Emotion Recognition GUI
After=graphical-session.target

[Service]
Type=simple
User=pi
Environment=DISPLAY=:0
WorkingDirectory=/home/pi/emotion-recognition
ExecStart=/usr/bin/python3 run_gui.py
Restart=always

[Install]
WantedBy=graphical-session.target
EOF

# Enable service
sudo systemctl enable emotion-recognition.service
```

## 📊 Performance Benchmarks

### Raspberry Pi 4 (4GB)

- **Resolution**: 640x480
- **FPS**: 15-20
- **CPU Usage**: 60-80%
- **Memory**: 400-600MB
- **Model**: TF Lite

### Raspberry Pi 3B+

- **Resolution**: 480x360
- **FPS**: 10-15
- **CPU Usage**: 80-95%
- **Memory**: 300-500MB
- **Model**: Quantized TF Lite

### Pi Zero 2W

- **Resolution**: 320x240
- **FPS**: 5-8
- **CPU Usage**: 90-100%
- **Memory**: 200-300MB
- **Model**: Basic detection

## 🎨 Customization

### Themes

Edit the GUI colors in `gui/emotion_gui_raspberry.py`:

```python
# Dark theme (default)
self.root.configure(bg='#2c3e50')

# Light theme
self.root.configure(bg='#ecf0f1')
```

### Emotions

Add or remove emotions by editing the emotions list:

```python
self.emotions = ['Happy', 'Sad', 'Angry']  # Simplified set
```

### Camera Settings

Adjust camera parameters:

```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Lower resolution
self.cap.set(cv2.CAP_PROP_FPS, 10)            # Lower FPS
```

## 📚 Additional Resources

- [Raspberry Pi Camera Guide](https://www.raspberrypi.org/documentation/usage/camera/)
- [OpenCV on Pi](https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/)
- [TensorFlow Lite for Pi](https://www.tensorflow.org/lite/guide/python)

## 🆘 Support

If you encounter issues:

1. Check the system log in the GUI
2. Run `python3 scripts/fear_disgust_diagnostic.py` for analysis
3. Verify camera with `raspistill -o test.jpg`
4. Check system resources with `htop`

## 🎯 Next Steps

1. **Collect more data**: Use `scripts/dataset_enhancement_guide.py`
2. **Optimize models**: Run `models/optimize_models.py`
3. **Train custom model**: Use `scripts/au_enhanced_training.py`
4. **Deploy to multiple Pis**: Create a distributed system
