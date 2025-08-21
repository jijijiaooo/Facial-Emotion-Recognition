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
        
        # Debug mode for Action Units
        self.debug_mode = False
        self.last_aus = {}
    
    def load_model(self):
        """Load emotion recognition model (RAF-DB preferred)"""
        # Try Hybrid AU models first (highest performance)
        hybrid_paths = [
            'models/hybrid_au_end_to_end_best.h5',
            'models/hybrid_au_hybrid_best.h5',
            '../models/hybrid_au_end_to_end_best.h5',
            '../models/hybrid_au_hybrid_best.h5'
        ]
        
        # RAF-DB models (good performance)
        raf_db_paths = [
            'models/raf_db_enhanced_best.h5',
            'models/raf_db_efficient_best.h5',
            'models/raf_db_resnet_best.h5',
            '../models/raf_db_enhanced_best.h5',
            '../models/raf_db_efficient_best.h5'
        ]
        
        # Also check for timestamped models (sorted by newest first)
        import glob
        hybrid_timestamped = sorted(glob.glob('models/hybrid_au_*_best.h5'), reverse=True)
        raf_db_timestamped = sorted(glob.glob('models/raf_db_*_best.h5'), reverse=True)
        
        hybrid_paths.extend(hybrid_timestamped)
        raf_db_paths.extend(raf_db_timestamped)
        
        # Fallback to original model
        original_paths = [
            'model_file_30epochs.h5',
            'models/model_file_30epochs.h5',
            '../models/model_file_30epochs.h5',
            '../../models/model_file_30epochs.h5'
        ]
        
        all_paths = hybrid_paths + raf_db_paths + original_paths
        
        for path in all_paths:
            if os.path.exists(path):
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(path)
                    if "hybrid_au" in path:
                        model_type = "Hybrid AU-CNN"
                    elif "raf_db" in path:
                        model_type = "RAF-DB"
                    else:
                        model_type = "Original"
                    print(f"{model_type} model loaded: {path}")
                    return model
                except ImportError:
                    print("TensorFlow not available")
                    break
                except Exception as e:
                    print(f"Model load error: {e}")
                    continue
        
        print("No model loaded - using basic detection")
        return None
    
    def init_face_detection(self):
        """Initialize face detection with multiple fallbacks"""
        # Try built-in OpenCV cascade first
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                print(f"Face detection loaded: built-in OpenCV")
                return cascade
        except Exception as e:
            print(f"Built-in cascade failed: {e}")
        
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
                        print(f"Face detection loaded: {path}")
                        return cascade
                except Exception as e:
                    print(f"Cascade {path} failed: {e}")
                    continue
        
        print("No face detection available - will use full frame")
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
            
            # Convert faces to list and check if empty
            faces_list = list(faces) if len(faces) > 0 else []
            
            if not faces_list:
                # Fallback to center region if no faces detected
                h, w = frame.shape[:2]
                margin_w = int(w * 0.25)
                margin_h = int(h * 0.25)
                return [(margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)]
            
            return faces_list
            
        except Exception as e:
            print(f"Face detection error: {e}")
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
            print(f"Model prediction error: {e}")
            return self.predict_basic(face_img)
    
    def extract_action_units(self, face_img):
        """Extract Action Units (AUs) from face image"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Ensure minimum face size for reliable analysis
            if h < 50 or w < 50:
                return {}
            
            # Define facial regions based on Action Units (more precise)
            regions = {
                'upper_face': gray[:h//2, :],  # Eyes, eyebrows, forehead
                'eye_region': gray[h//5:h//2, :],  # Eye area (adjusted)
                'eyebrow_region': gray[:h//3, :],  # Eyebrow area (larger)
                'nose_region': gray[h//3:2*h//3, w//3:2*w//3],  # Nose area (centered)
                'mouth_region': gray[2*h//3:, :],  # Mouth area
                'cheek_left': gray[h//3:2*h//3, :w//2],  # Left cheek (larger)
                'cheek_right': gray[h//3:2*h//3, w//2:],  # Right cheek (larger)
                'jaw_region': gray[3*h//4:, :]  # Jaw/chin area
            }
            
            # Calculate Action Unit features (improved)
            aus = {}
            
            # AU1 & AU2: Inner/Outer Brow Raiser (surprise, fear)
            eyebrow_intensity = np.std(regions['eyebrow_region'])
            eyebrow_contrast = np.max(regions['eyebrow_region']) - np.min(regions['eyebrow_region'])
            aus['AU1_AU2'] = eyebrow_intensity + eyebrow_contrast / 5
            
            # AU4: Brow Lowerer (anger, concentration)
            brow_darkness = 255 - np.mean(regions['eyebrow_region'])
            brow_contrast = np.std(regions['eyebrow_region'])
            aus['AU4'] = brow_darkness + brow_contrast / 3
            
            # AU5: Upper Lid Raiser (surprise, fear)
            eye_openness = np.std(regions['eye_region'])
            eye_brightness = np.mean(regions['eye_region'])
            aus['AU5'] = eye_openness + eye_brightness / 10
            
            # AU6 & AU12: Cheek Raiser & Lip Corner Puller (happiness)
            cheek_activity = (np.std(regions['cheek_left']) + np.std(regions['cheek_right'])) / 2
            mouth_upper = regions['mouth_region'][:h//8, :] if h//8 > 0 else regions['mouth_region'][:1, :]
            mouth_brightness = np.mean(mouth_upper)
            aus['AU6_AU12'] = cheek_activity + mouth_brightness / 8
            
            # AU9: Nose Wrinkler (disgust)
            nose_wrinkles = np.std(regions['nose_region'])
            nose_contrast = np.max(regions['nose_region']) - np.min(regions['nose_region'])
            aus['AU9'] = nose_wrinkles + nose_contrast / 8
            
            # AU10: Upper Lip Raiser (disgust)
            upper_lip = regions['mouth_region'][:h//10, :] if h//10 > 0 else regions['mouth_region'][:1, :]
            aus['AU10'] = np.std(upper_lip) + (255 - np.mean(upper_lip)) / 10
            
            # AU15: Lip Corner Depressor (sadness)
            lower_mouth = regions['mouth_region'][h//8:, :] if h//8 > 0 else regions['mouth_region']
            mouth_darkness = 255 - np.mean(lower_mouth)
            mouth_variation = np.std(lower_mouth)
            aus['AU15'] = mouth_darkness + mouth_variation / 5
            
            # AU20: Lip Stretcher (fear)
            mouth_width_activity = np.std(regions['mouth_region'], axis=1)
            mouth_horizontal = np.std(regions['mouth_region'], axis=0)
            aus['AU20'] = np.mean(mouth_width_activity) + np.mean(mouth_horizontal) / 5
            
            # AU25: Lips Part (surprise, fear)
            mouth_center = regions['mouth_region'][h//12:h//6, :] if h//12 > 0 else regions['mouth_region']
            mouth_opening = np.std(mouth_center)
            aus['AU25'] = mouth_opening + (255 - np.mean(mouth_center)) / 8
            
            # AU26: Jaw Drop (surprise)
            jaw_drop = np.mean(regions['jaw_region'])
            jaw_contrast = np.std(regions['jaw_region'])
            aus['AU26'] = (255 - jaw_drop) + jaw_contrast / 5
            
            return aus
            
        except Exception as e:
            print(f"AU extraction error: {e}")
            return {}
    
    def predict_basic(self, face_img):
        """Enhanced emotion prediction using Action Units"""
        try:
            # Extract Action Units
            aus = self.extract_action_units(face_img)
            
            if not aus:
                return "Neutral", 0.5
            
            # Store for debug display
            self.last_aus = aus
            
            # Convert to grayscale for additional features
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Emotion classification based on Action Units (balanced for all emotions)
            emotion_scores = {
                'Happy': 0,
                'Sad': 0,
                'Angry': 0,
                'Fear': 0,
                'Surprise': 0,
                'Disgust': 0,
                'Neutral': 0.15  # Lower baseline to allow other emotions
            }
            
            # Happy: AU6+AU12 (cheek raiser + lip corner puller)
            happy_score = 0
            if aus.get('AU6_AU12', 0) > 25:  # Slightly lower threshold
                happy_score += 0.6
            if brightness > 115:
                happy_score += 0.2
            if happy_score > 0.25:  # Lower threshold
                emotion_scores['Happy'] = happy_score
            
            # Sad: AU15 (lip corner depressor) + additional sad indicators
            sad_score = 0
            if aus.get('AU15', 0) > 18:  # Lower threshold
                sad_score += 0.5
            if brightness < 125:
                sad_score += 0.2
            # Additional sad indicators
            if aus.get('AU1_AU2', 0) > 10 and aus.get('AU4', 0) > 10:  # Lower thresholds
                sad_score += 0.2
            if contrast < 32:
                sad_score += 0.1
            if aus.get('AU15', 0) > 12:  # Lower threshold
                sad_score += 0.2
            if sad_score > 0.25:  # Lower threshold
                emotion_scores['Sad'] = sad_score
            
            # Angry: AU4 (brow lowerer) + contrast (more sensitive)
            angry_score = 0
            if aus.get('AU4', 0) > 25:  # Much lower threshold
                angry_score += 0.5
            if contrast > 25:  # Lower contrast requirement
                angry_score += 0.3
            # Additional angry indicators
            if aus.get('AU4', 0) > 18 and brightness < 120:  # More sensitive
                angry_score += 0.3
            # Check for tense facial features
            if aus.get('AU4', 0) > 15:  # Any brow lowering
                angry_score += 0.2
            if angry_score > 0.25:  # Much lower threshold
                emotion_scores['Angry'] = angry_score
            
            # Fear: AU1+AU2 (brow raiser) + AU5 (upper lid raiser) + AU20 (lip stretcher)
            fear_score = 0
            if aus.get('AU1_AU2', 0) > 20:  # Lower threshold
                fear_score += 0.3
            if aus.get('AU5', 0) > 18:  # Lower threshold
                fear_score += 0.3
            if aus.get('AU20', 0) > 15:  # Lower threshold
                fear_score += 0.3
            # Additional fear indicators
            if brightness > 120 and contrast > 30:  # More sensitive
                fear_score += 0.2
            # Wide eyes indicator
            if aus.get('AU5', 0) > 12:  # Any upper lid raising
                fear_score += 0.2
            if fear_score > 0.25:  # Much lower threshold
                emotion_scores['Fear'] = fear_score
            
            # Surprise: AU1+AU2 (brow raiser) + AU5 (upper lid raiser) + AU25+AU26 (jaw drop)
            surprise_score = 0
            if aus.get('AU1_AU2', 0) > 25:  # Lower threshold
                surprise_score += 0.3
            if aus.get('AU5', 0) > 22:  # Lower threshold
                surprise_score += 0.3
            if aus.get('AU25', 0) > 15 or aus.get('AU26', 0) > 14:  # Lower thresholds
                surprise_score += 0.4
            if surprise_score > 0.3:  # Lower threshold
                emotion_scores['Surprise'] = surprise_score
            
            # Disgust: AU9 (nose wrinkler) + AU10 (upper lip raiser) - MUCH more sensitive
            disgust_score = 0
            if aus.get('AU9', 0) > 12:  # Much lower threshold
                disgust_score += 0.4
            if aus.get('AU10', 0) > 10:  # Much lower threshold
                disgust_score += 0.4
            # Additional disgust indicators (more sensitive)
            if aus.get('AU9', 0) > 8 or aus.get('AU10', 0) > 8:  # Any nose/lip activity
                disgust_score += 0.3
            # Check for upper lip curl
            if aus.get('AU10', 0) > 5:  # Very sensitive
                disgust_score += 0.2
            if disgust_score > 0.2:  # Much lower threshold
                emotion_scores['Disgust'] = disgust_score
            
            # Conservative fallback system - only when no clear emotion detected
            max_score = max(emotion_scores.values())
            if max_score < 0.35:  # Only use fallback when really unclear
                # Simple fallback based on brightness and contrast patterns
                h, w = gray.shape
                
                # Analyze different regions
                upper_region = gray[:h//3, :]
                middle_region = gray[h//3:2*h//3, :]
                lower_region = gray[2*h//3:, :]
                
                upper_brightness = np.mean(upper_region)
                middle_brightness = np.mean(middle_region)
                lower_brightness = np.mean(lower_region)
                
                # Conservative pattern matching - only clear cases
                if lower_brightness < middle_brightness * 0.8 and brightness < 120:  # Very dark mouth
                    emotion_scores['Sad'] = max(emotion_scores['Sad'], 0.35)
                elif lower_brightness > middle_brightness * 1.15 and brightness > 120:  # Very bright mouth
                    emotion_scores['Happy'] = max(emotion_scores['Happy'], 0.35)
                elif upper_brightness > brightness * 1.15 and contrast > 40:  # Very bright upper region
                    emotion_scores['Surprise'] = max(emotion_scores['Surprise'], 0.35)
                elif contrast > 45 and brightness < 105:  # Very high contrast, very dark
                    emotion_scores['Angry'] = max(emotion_scores['Angry'], 0.35)
            
            # Find the emotion with highest score
            predicted_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(0.95, emotion_scores[predicted_emotion])
            
            # Debug output (uncomment for debugging)
            # if hasattr(self, 'debug_mode') and self.debug_mode:
            #     print(f"AU Values: {aus}")
            #     print(f"Emotion Scores: {emotion_scores}")
            #     print(f"Predicted: {predicted_emotion} ({confidence:.2f})")
            
            # Boost confidence for clear detections
            if confidence > 0.4:
                confidence = min(0.9, confidence + 0.1)
            
            return predicted_emotion, confidence
                
        except Exception as e:
            print(f"Enhanced prediction error: {e}")
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
            print("Could not open camera")
            return
        
        print("Starting Enhanced Emotion Detection with Action Units")
        print("Press 'q' to quit, 's' to save screenshot, 'd' to toggle debug mode")
        
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
                    
                    # Debug mode: show Action Units
                    if self.debug_mode and self.last_aus:
                        debug_y = y + h + 20
                        for i, (au_name, au_value) in enumerate(self.last_aus.items()):
                            debug_text = f"{au_name}: {au_value:.1f}"
                            cv2.putText(frame, debug_text, (x, debug_y + i*15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
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
                    print("Screenshot saved as emotion_screenshot.jpg")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Simple emotion detection stopped")

def main():
    print("Simple Emotion Detection")
    print("=" * 30)
    print("This version works without external cascade files")
    print("and provides basic emotion recognition")
    
    detector = SimpleEmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()