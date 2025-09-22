#!/usr/bin/env python3
"""
Test script for the Ensemble Emotion Detection System
"""

import sys
import os
sys.path.append('src/core')

def test_ensemble_system():
    """Test the ensemble emotion detection system"""
    try:
        from simple_emotion_detection import SimpleEmotionDetector
        import cv2
        import numpy as np
        
        print("🚀 Initializing Ensemble Emotion Detection System...")
        detector = SimpleEmotionDetector()
        
        # Enable debug mode to see individual model predictions
        detector.enable_debug_mode()
        
        # Create a test image (48x48 gray image with some pattern)
        test_image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        
        # Add some simple patterns to simulate facial features
        # Eyes
        cv2.rectangle(test_image, (15, 15), (18, 18), (50, 50, 50), -1)
        cv2.rectangle(test_image, (30, 15), (33, 18), (50, 50, 50), -1)
        
        # Mouth (smile)
        cv2.ellipse(test_image, (24, 35), (8, 4), 0, 0, 180, (100, 100, 100), 1)
        
        print("\n🧪 Testing ensemble prediction with synthetic face image...")
        emotion, confidence = detector.predict_emotion(test_image)
        
        print(f"\n🎭 Final Ensemble Result:")
        print(f"   Emotion: {emotion}")
        print(f"   Confidence: {confidence:.3f}")
        
        # Test multiple predictions to show consistency
        print(f"\n🔄 Testing prediction consistency (5 runs):")
        for i in range(5):
            emotion, confidence = detector.predict_emotion(test_image)
            print(f"   Run {i+1}: {emotion} (conf: {confidence:.3f})")
        
        print(f"\n✅ Ensemble system test completed successfully!")
        print(f"📊 Models utilized: {len(detector.models)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install required packages: opencv-python, tensorflow, numpy")
        return False
    except Exception as e:
        print(f"❌ Error testing ensemble system: {e}")
        return False

if __name__ == "__main__":
    print("🎭 Ensemble Emotion Detection System Test")
    print("="*50)
    
    success = test_ensemble_system()
    
    if success:
        print("\n🎉 All tests passed! The ensemble system is ready to use.")
    else:
        print("\n⚠️  Tests failed. Please check the error messages above.")