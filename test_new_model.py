#!/usr/bin/env python3
"""
Test script to demonstrate the new Simple CNN model
This shows how the model is automatically loaded and used
"""

import sys
sys.path.insert(0, 'src/core')
from simple_emotion_detection import SimpleEmotionDetector
import cv2
import numpy as np

def test_model_info():
    """Display information about the loaded model"""
    print("\n" + "="*70)
    print("EMOTION DETECTION MODEL INFO")
    print("="*70)
    
    detector = SimpleEmotionDetector()
    
    print(f"\n✅ Models loaded: {len(detector.models)}")
    for i, model_info in enumerate(detector.models):
        print(f"\nModel {i+1}:")
        print(f"  Type: {model_info['type']}")
        print(f"  Path: {model_info['path']}")
        print(f"  Input shape: {model_info['input_shape']}")
    
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)
    
    # Create some test images with different random patterns
    test_cases = [
        ("Random noise", np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)),
        ("Dark image", np.random.randint(0, 50, (96, 96, 3), dtype=np.uint8)),
        ("Bright image", np.random.randint(200, 255, (96, 96, 3), dtype=np.uint8)),
    ]
    
    for name, test_image in test_cases:
        emotion, confidence = detector.predict_emotion(test_image)
        print(f"\n{name}:")
        print(f"  Predicted: {emotion}")
        print(f"  Confidence: {confidence:.4f}")
    
    print("\n" + "="*70)
    print("To test with webcam, run:")
    print("  python3 src/core/simple_emotion_detection.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_model_info()
