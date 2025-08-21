#!/usr/bin/env python3
"""
Simple Emotion Detection - No External Dependencies
Works without Haar cascade files or complex models
"""

import cv2
import numpy as np
import os
import sys

class SimpleEmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (255, 0, 255),
            'Happy': (0, 255, 255), 'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
            'Surprise': (0, 165, 255)
        }
        
        # Try to load model
        self.model = self.load_model()
        
        # Initialize face detection
        self.face_cascade = self.init_face_detection()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start = cv2.getTickCount()
        self.current_fps = 0
    
    def load_model(self):
        """Load emotion recognition model"""
        model_paths = [
            'model_file_30epochs.h5',
            'models/model_file_30epochs.h5',
            '../models/model_file_30epochs.h5',
            '../../models/model_file_30epochs.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(path)
                    print(f"✅ Model loaded: {path}")
                    return model
                except ImportError:
                    print("⚠️ TensorFlow not available")
                    break
                except Exception as e:
                    print(f"⚠️ Model load error: {e}")
                    continue
        
        print("⚠️ No model loaded - using basic detection")
        return None
    
    def init_face_detection(self):
        """Initialize face detection with multiple fallbacks"""
        # Try built-in OpenCV cascade first
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                print(f"✅ Face detection loaded: built-in OpenCV")
                return cascade
        except Exception as e:
            print(f"⚠️ Built-in cascade failed: {e}")
        
        # Try local cascade files
        cascade_paths = [
            'haarcascade_frontalface_default.xml',
            'config/haarcascade_frontalface_default.xml',
            '../config/haarcascade_frontalface_default.xml',
            '../../config/haarcascade_frontalface_default.xml'
        ]
        
        for path in cascade_paths:
            if os.path.exists(path):
                try:
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        print(f"✅ Face detection loaded: {path}")
                        return cascade
                except Exception as e:
                    print(f"⚠️ Cascade {path} failed: {e}")
                    continue
        
        print("❌ No face detection available - will use full frame")
        return None
    
    def detect_faces(self, frame):
        """Detect faces with fallback to full frame"""
        if self.face_cascade is None:
            # Use full frame as "face"
            h, w = frame.shape[:2]
            # Use center 60% of frame
            margin_w = int(w * 0.2)
            margin_h = int(h * 0.2)
            return [(margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)]
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                # Fallback to center region if no faces detected
                h, w = frame.shape[:2]
                margin_w = int(w * 0.25)
                margin_h = int(h * 0.25)
                return [(margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)]
            
            return faces
            
        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
            # Fallback to center region
            h, w = frame.shape[:2]
            margin_w = int(w * 0.25)
            margin_h = int(h * 0.25)
            return [(margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)]
    
    def predict_emotion(self, face_img):
        """Predict emotion with model or basic rules"""
        if self.model is not None:
            return self.predict_with_model(face_img)
        else:
            return self.predict_basic(face_img)
    
    def predict_with_model(self, face_img):
        """Predict using trained model"""
        try:
            # Preprocess
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 48, 48, 1))
            
            # Predict
            result = self.model.predict(reshaped, verbose=0)
            emotion_idx = np.argmax(result, axis=1)[0]
            confidence = result[0][emotion_idx]
            
            return self.emotions[emotion_idx], confidence
            
        except Exception as e:
            print(f"⚠️ Model prediction error: {e}")
            return self.predict_basic(face_img)
    
    def predict_basic(self, face_img):
        """Basic emotion prediction using image analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Basic features
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Divide face into regions
            h, w = gray.shape
            
            # Upper region (eyes/eyebrows)
            upper_region = gray[:h//3, :]
            upper_brightness = np.mean(upper_region)
            
            # Middle region (nose)
            middle_region = gray[h//3:2*h//3, :]
            middle_brightness = np.mean(middle_region)
            
            # Lower region (mouth)
            lower_region = gray[2*h//3:, :]
            lower_brightness = np.mean(lower_region)
            
            # Simple rules based on brightness patterns
            confidence = 0.6  # Base confidence for rule-based
            
            # Happy: mouth region brighter (smile)
            if lower_brightness > middle_brightness * 1.1 and brightness > 120:
                return "Happy", min(0.8, confidence + 0.2)
            
            # Sad: overall darker, mouth darker
            elif brightness < 100 and lower_brightness < middle_brightness * 0.9:
                return "Sad", confidence
            
            # Surprise: high contrast, bright upper region
            elif contrast > 40 and upper_brightness > brightness * 1.1:
                return "Surprise", confidence
            
            # Angry: high contrast, darker overall
            elif contrast > 35 and brightness < 110:
                return "Angry", confidence
            
            # Fear: similar to surprise but different brightness pattern
            elif contrast > 30 and upper_brightness > brightness * 1.05:
                return "Fear", confidence - 0.1
            
            # Disgust: middle region (nose) analysis
            elif middle_brightness < brightness * 0.95 and contrast > 25:
                return "Disgust", confidence - 0.1
            
            # Default to neutral
            else:
                return "Neutral", confidence - 0.1
                
        except Exception as e:
            print(f"⚠️ Basic prediction error: {e}")
            return "Neutral", 0.5
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.fps_start) / cv2.getTickFrequency()
        
        if time_diff >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start = current_time
    
    def run(self):
        """Main detection loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("🎭 Starting Simple Emotion Detection")
        print("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.update_fps()
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Predict emotion
                    emotion, confidence = self.predict_emotion(face_img)
                    
                    # Draw results
                    color = self.colors.get(emotion, (255, 255, 255))
                    
                    # Face rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Emotion label
                    label = f"{emotion}"
                    if confidence > 0.6:
                        label += f" ({confidence:.2f})"
                    
                    # Text background
                    cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                    cv2.putText(frame, label, (x+5, y-8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Show FPS
                fps_text = f"FPS: {self.current_fps}"
                cv2.putText(frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Simple Emotion Detection", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('emotion_screenshot.jpg', frame)
                    print("📸 Screenshot saved as emotion_screenshot.jpg")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Simple emotion detection stopped")

def main():
    print("🎭 Simple Emotion Detection")
    print("=" * 30)
    print("This version works without external cascade files")
    print("and provides basic emotion recognition")
    
    detector = SimpleEmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()