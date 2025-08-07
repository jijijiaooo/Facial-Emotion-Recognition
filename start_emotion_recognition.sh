#!/bin/bash
# Emotion Recognition Startup Script

# Wait for desktop to load
sleep 10

# Launch emotion recognition GUI
cd /Users/jiaoshihlo/Codes/Facial Emotion Recognition
python3 run_gui.py

# Log any errors
echo "Emotion Recognition started at $(date)" >> /tmp/emotion_recognition.log
