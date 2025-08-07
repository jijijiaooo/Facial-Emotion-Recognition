#!/usr/bin/env python3
"""
Quick Fix for Fear and Disgust Detection
Implements immediate improvements to boost these emotion classes
"""

import cv2
import numpy as np
import os
from collections import Counter

class FearDisgustFixer:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Enhanced preprocessing for fear and disgust
        self.fear_enhancer = self.create_fear_enhancer()
        self.disgust_enhancer = self.create_disgust_enhancer()
        
        # Class weights to boost underrepresented emotions
        self.class_weights = {
            0: 1.0,  # Angry
            1: 3.0,  # Disgust (boost heavily)
            2: 3.0,  # Fear (boost heavily)
            3: 1.0,  # Happy
            4: 1.0,  # Neutral
            5: 1.0,  # Sad
            6: 1.0   # Surprise
        }
    
    def create_fear_enhancer(self):
        """Create preprocessing pipeline optimized for fear detection"""
        def enhance_fear_features(img):
            # Fear has wide eyes and raised eyebrows - enhance these regions
            h, w = img.shape
            
            # Enhance upper region (eyebrows/forehead)
            upper_region = img[:h//3, :]
            enhanced_upper = cv2.equalizeHist(upper_region)
            
            # Enhance middle region (eyes)
            middle_region = img[h//3:2*h//3, :]
            enhanced_middle = cv2.equalizeHist(middle_region)
            
            # Keep lower region normal
            lower_region = img[2*h//3:, :]
            
            # Combine regions
            enhanced = np.vstack([enhanced_upper, enhanced_middle, lower_region])
            
            # Slight edge enhancement for eyebrow lines
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and enhanced
            result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            return result
        
        return enhance_fear_features
    
    def create_disgust_enhancer(self):
        """Create preprocessing pipeline optimized for disgust detection"""
        def enhance_disgust_features(img):
            # Disgust has wrinkled nose - enhance middle region texture
            h, w = img.shape
            
            # Keep upper region normal
            upper_region = img[:h//3, :]
            
            # Enhance middle region (nose area) for wrinkles
            middle_region = img[h//3:2*h//3, :]
            
            # Apply unsharp masking to enhance nose wrinkles
            blurred = cv2.GaussianBlur(middle_region, (3, 3), 0)
            enhanced_middle = cv2.addWeighted(middle_region, 1.5, blurred, -0.5, 0)
            
            # Enhance lower region (mouth area)
            lower_region = img[2*h//3:, :]
            enhanced_lower = cv2.equalizeHist(lower_region)
            
            # Combine regions
            enhanced = np.vstack([upper_region, enhanced_middle, enhanced_lower])
            
            return enhanced
        
        return enhance_disgust_features
    
    def enhanced_preprocessing(self, face_img, predicted_emotion=None):
        """Apply emotion-specific preprocessing"""
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # Resize to model input size
        resized = cv2.resize(gray, (48, 48))
        
        # Apply emotion-specific enhancement
        if predicted_emotion == 'Fear' or predicted_emotion is None:
            fear_enhanced = self.fear_enhancer(resized)
        else:
            fear_enhanced = resized
        
        if predicted_emotion == 'Disgust' or predicted_emotion is None:
            disgust_enhanced = self.disgust_enhancer(resized)
        else:
            disgust_enhanced = resized
        
        # Return multiple versions for ensemble
        return {
            'original': resized / 255.0,
            'fear_enhanced': fear_enhanced / 255.0,
            'disgust_enhanced': disgust_enhanced / 255.0
        }
    
    def ensemble_prediction_with_boost(self, face_img, model):
        """Make prediction with fear/disgust boosting"""
        try:
            # Get multiple preprocessed versions
            processed_versions = self.enhanced_preprocessing(face_img)
            
            predictions = []
            
            # Predict with each version
            for version_name, processed_img in processed_versions.items():
                input_data = processed_img.reshape(1, 48, 48, 1)
                pred = model.predict(input_data, verbose=0)[0]
                predictions.append(pred)
            
            # Average predictions
            avg_prediction = np.mean(predictions, axis=0)
            
            # Apply class weights to boost fear and disgust
            boosted_prediction = avg_prediction.copy()
            boosted_prediction[1] *= self.class_weights[1]  # Boost disgust
            boosted_prediction[2] *= self.class_weights[2]  # Boost fear
            
            # Renormalize
            boosted_prediction = boosted_prediction / np.sum(boosted_prediction)
            
            # Get final prediction
            emotion_idx = np.argmax(boosted_prediction)
            confidence = boosted_prediction[emotion_idx]
            
            return self.emotions[emotion_idx], confidence, {
                'original_pred': avg_prediction,
                'boosted_pred': boosted_prediction,
                'top_3': [(self.emotions[i], boosted_prediction[i]) 
                         for i in np.argsort(boosted_prediction)[-3:][::-1]]
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Neutral", 0.5, {}
    
    def analyze_prediction_details(self, debug_info):
        """Analyze prediction details for debugging"""
        if not debug_info:
            return
        
        print(f"Original prediction: {debug_info['original_pred']}")
        print(f"Boosted prediction: {debug_info['boosted_pred']}")
        print(f"Top 3 predictions:")
        for emotion, conf in debug_info['top_3']:
            print(f"  {emotion}: {conf:.3f}")

class EnhancedEmotionDetector:
    """Enhanced detector with fear/disgust fixes"""
    
    def __init__(self):
        self.fixer = FearDisgustFixer()
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (255, 0, 255),
            'Happy': (0, 255, 255), 'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
            'Surprise': (0, 165, 255)
        }
        
        # Load model
        self.model = None
        self.load_model()
        
        # Initialize face detection
        self.init_face_detection()
        
        # Performance tracking
        self.emotion_counts = Counter()
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
    
    def load_model(self):
        """Load the emotion recognition model"""
        try:
            from tensorflow import keras
            self.model = keras.models.load_model('model_file_30epochs.h5')
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            return False
        return True
    
    def init_face_detection(self):
        """Initialize face detection"""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.7
            )
            self.detection_method = "mediapipe"
            print("✅ MediaPipe face detection initialized")
        except ImportError:
            # Fallback to OpenCV
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.detection_method = "opencv"
            print("✅ OpenCV face detection initialized")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        if self.detection_method == "mediapipe":
            return self.detect_faces_mediapipe(frame)
        else:
            return self.detect_faces_opencv(frame)
    
    def detect_faces_mediapipe(self, frame):
        """MediaPipe face detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(w - x, int(bbox.width * w))
                height = min(h - y, int(bbox.height * h))
                
                if width > 30 and height > 30:
                    faces.append((x, y, width, height))
        
        return faces
    
    def detect_faces_opencv(self, frame):
        """OpenCV face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start = time.time()
    
    def run(self):
        """Main detection loop with fear/disgust fixes"""
        if not self.model:
            print("❌ No model loaded, cannot run detection")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("🚀 Starting Enhanced Fear/Disgust Detection")
        print("🎯 Boosted detection for Fear and Disgust emotions")
        print("Press 'q' to quit, 'd' for debug info, 's' for statistics")
        
        show_debug = False
        show_stats = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.update_fps()
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process each face
                for face_data in faces:
                    x, y, w, h = face_data[:4]
                    
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.size > 0:
                        # Enhanced prediction with fear/disgust boost
                        emotion, confidence, debug_info = self.fixer.ensemble_prediction_with_boost(
                            face_img, self.model
                        )
                        
                        # Track emotion counts
                        self.emotion_counts[emotion] += 1
                        
                        # Draw results
                        color = self.colors.get(emotion, (255, 255, 255))
                        
                        # Highlight fear and disgust with thicker borders
                        thickness = 4 if emotion in ['Fear', 'Disgust'] else 2
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                        
                        # Enhanced label for fear/disgust
                        label = f"{emotion.upper()}"
                        if emotion in ['Fear', 'Disgust']:
                            label += f" ⚡"  # Special indicator
                        if confidence > 0.6:
                            label += f" ({confidence:.2f})"
                        
                        # Text background
                        cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
                        cv2.putText(frame, label, (x+5, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        # Show debug info if requested
                        if show_debug and emotion in ['Fear', 'Disgust']:
                            print(f"\n{emotion} detected:")
                            self.fixer.analyze_prediction_details(debug_info)
                
                # Show statistics
                if show_stats:
                    stats_text = f"FPS: {self.current_fps} | Fear: {self.emotion_counts['Fear']} | Disgust: {self.emotion_counts['Disgust']}"
                    cv2.putText(frame, stats_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Enhanced Fear/Disgust Detection", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    show_debug = not show_debug
                    print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
                elif key == ord('s'):
                    show_stats = not show_stats
                    print(f"Statistics: {'ON' if show_stats else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Show final statistics
            print(f"\n📊 Final Emotion Detection Statistics:")
            total_detections = sum(self.emotion_counts.values())
            if total_detections > 0:
                for emotion in self.emotions:
                    count = self.emotion_counts[emotion]
                    percentage = (count / total_detections) * 100
                    print(f"{emotion:10}: {count:4d} ({percentage:5.1f}%)")
                
                fear_percent = (self.emotion_counts['Fear'] / total_detections) * 100
                disgust_percent = (self.emotion_counts['Disgust'] / total_detections) * 100
                
                print(f"\n🎯 Target Emotions:")
                print(f"Fear detection rate: {fear_percent:.1f}%")
                print(f"Disgust detection rate: {disgust_percent:.1f}%")
                
                if fear_percent > 5 and disgust_percent > 5:
                    print("✅ Good detection rates for fear and disgust!")
                else:
                    print("⚠️ Still low detection rates - consider dataset improvements")

def main():
    import time
    
    print("🎯 Fear and Disgust Detection Fix")
    print("=" * 40)
    print("This enhanced detector specifically boosts fear and disgust recognition")
    
    detector = EnhancedEmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()