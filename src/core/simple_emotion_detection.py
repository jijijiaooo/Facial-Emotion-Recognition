#!/usr/bin/env python3
"""
Simple Emotion Detection - No External Dependencies
Works without Haar cascade files or complex models
"""

import cv2
import numpy as np
import os
import sys
import pickle

class SimpleEmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (255, 0, 255),
            'Happy': (0, 255, 255), 'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
            'Surprise': (0, 165, 255)
        }
        self.models = self.load_all_models()
        self.face_cascade = self.init_face_detection()
        self.fps_counter = 0
        self.fps_start = cv2.getTickCount()
        self.current_fps = 0
        self.debug_mode = False
        self.last_aus = {}

    def load_all_models(self):
        """Load all available models for ensembling."""
        model_files = [
            'models/ensemble_raf_db1_20250904_124505.pkl',
            'models/raf_db_simple_cnn.h5',
            'models/raf_db1_custom_20250904_113432_best.h5',
            'models/raf_db1_custom_20250904_113432_final.h5',
            'models/raf_db1_mobilenet_20250904_001856_best.h5',
            'models/raf_db1_mobilenet_20250904_001856_final.h5',
            'models/raf_db1_mobilenet_20250904_001856_finetuned_best.h5',
            'models/raf_db1_resnet_20250902_221544_best.h5',
            'models/raf_db1_resnet_20250903_213606_best.h5',
            'models/raf_db1_resnet_20250903_213606_final.h5',
            'models/raf_db1_resnet_20250903_213606_finetuned_best.h5'
        ]
        loaded_models = []
        for model_path in model_files:
            if model_path.endswith('.h5') and os.path.exists(model_path):
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(model_path)
                    loaded_models.append(('keras', model))
                    print(f"Loaded Keras model: {model_path}")
                except Exception as e:
                    print(f"Failed to load Keras model {model_path}: {e}")
            elif model_path.endswith('.pkl') and os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    loaded_models.append(('sklearn', model))
                    print(f"Loaded pickle model: {model_path}")
                except Exception as e:
                    print(f"Failed to load pickle model {model_path}: {e}")
        if not loaded_models:
            print("No models loaded - using basic detection")
        return loaded_models

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
    
    def preprocess_face(self, face_img):
        """Resize and normalize face image for model input."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        return reshaped

    def predict_emotion(self, face_img):
        """Predict emotion using ensemble of models or fallback to basic rules."""
        if self.models:
            votes = []
            probs = []
            for model_type, model in self.models:
                try:
                    img = self.preprocess_face(face_img)
                    if model_type == 'keras':
                        pred = model.predict(img, verbose=0)
                        idx = int(np.argmax(pred))
                        votes.append(idx)
                        probs.append(pred[0])
                    elif model_type == 'sklearn':
                        flat_img = img.flatten().reshape(1, -1)
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(flat_img)
                            idx = int(np.argmax(pred))
                            votes.append(idx)
                            probs.append(pred[0])
                        else:
                            idx = int(model.predict(flat_img)[0])
                            votes.append(idx)
                except Exception as e:
                    print(f"Model prediction error: {e}")
            if votes:
                # Majority vote
                final_idx = int(np.bincount(votes).argmax())
                # Average confidence if available
                confidence = float(np.mean([p[final_idx] for p in probs if len(p) > final_idx])) if probs else 1.0
                emotion = self.emotions[final_idx]
                return emotion, confidence
        # Fallback
        return self.predict_basic(face_img)
    
    def extract_action_units(self, face_img):
        """Improved Action Unit (AU) extraction from face image"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Ensure minimum face size for reliable analysis
            if h < 50 or w < 50:
                return {}

            # More precise facial regions
            regions = {
                'forehead': gray[:h//6, w//4:3*w//4],  # Forehead
                'eyebrow_left': gray[h//12:h//4, w//10:w//2],  # Left eyebrow
                'eyebrow_right': gray[h//12:h//4, w//2:9*w//10],  # Right eyebrow
                'eye_left': gray[h//4:h//2, w//10:w//2],  # Left eye
                'eye_right': gray[h//4:h//2, w//2:9*w//10],  # Right eye
                'nose': gray[h//3:2*h//3, w//3:2*w//3],  # Nose
                'mouth_upper': gray[2*h//3:2*h//3+h//12, w//4:3*w//4],  # Upper lip
                'mouth_lower': gray[2*h//3+h//12:h, w//4:3*w//4],  # Lower lip
                'mouth': gray[2*h//3:h, w//4:3*w//4],  # Mouth region
                'jaw': gray[5*h//6:h, w//4:3*w//4],  # Jaw/chin
            }

            aus = {}

            # AU1/AU2: Brow raiser (eyebrow std + mean diff from forehead)
            brow_left_mean = np.mean(regions['eyebrow_left'])
            brow_right_mean = np.mean(regions['eyebrow_right'])
            forehead_mean = np.mean(regions['forehead'])
            brow_raise = ((brow_left_mean + brow_right_mean) / 2) - forehead_mean
            brow_std = (np.std(regions['eyebrow_left']) + np.std(regions['eyebrow_right'])) / 2
            aus['AU1_AU2'] = brow_std + brow_raise / 4

            # AU4: Brow lowerer (darker, less contrast)
            brow_lower = 255 - ((brow_left_mean + brow_right_mean) / 2)
            aus['AU4'] = brow_lower + brow_std / 3

            # AU5: Upper lid raiser (eye openness)
            eye_openness = (np.std(regions['eye_left']) + np.std(regions['eye_right'])) / 2
            eye_brightness = (np.mean(regions['eye_left']) + np.mean(regions['eye_right'])) / 2
            aus['AU5'] = eye_openness + eye_brightness / 10

            # AU6/AU12: Cheek raiser & lip corner puller (smile)
            mouth_upper_mean = np.mean(regions['mouth_upper'])
            mouth_lower_mean = np.mean(regions['mouth_lower'])
            cheek_activity = abs(mouth_upper_mean - mouth_lower_mean)
            mouth_std = np.std(regions['mouth'])
            aus['AU6_AU12'] = cheek_activity + mouth_std / 8

            # AU9: Nose wrinkler (nose std)
            aus['AU9'] = np.std(regions['nose'])

            # AU10: Upper lip raiser (upper lip std + darkness)
            upper_lip_darkness = 255 - np.mean(regions['mouth_upper'])
            aus['AU10'] = np.std(regions['mouth_upper']) + upper_lip_darkness / 10

            # AU15: Lip corner depressor (lower lip darkness + std)
            lower_lip_darkness = 255 - np.mean(regions['mouth_lower'])
            aus['AU15'] = lower_lip_darkness + np.std(regions['mouth_lower']) / 5

            # AU20: Lip stretcher (mouth width std)
            mouth_width_std = np.std(regions['mouth'], axis=0)
            aus['AU20'] = np.mean(mouth_width_std)

            # AU25: Lips part (mouth opening: vertical gradient)
            vertical_profile = np.mean(regions['mouth'], axis=1)
            mouth_opening = np.max(vertical_profile) - np.min(vertical_profile)
            aus['AU25'] = mouth_opening

            # AU26: Jaw drop (jaw darkness + std)
            jaw_darkness = 255 - np.mean(regions['jaw'])
            jaw_std = np.std(regions['jaw'])
            aus['AU26'] = jaw_darkness + jaw_std / 5

            # Symmetry features (optional, for future use)
            # left_right_diff = abs(np.mean(regions['eyebrow_left']) - np.mean(regions['eyebrow_right']))
            # aus['brow_symmetry'] = left_right_diff

            # Normalize AUs by face intensity to reduce lighting effects
            face_mean = np.mean(gray)
            for k in aus:
                aus[k] = aus[k] / (face_mean + 1e-5) * 100

            return aus

        except Exception as e:
            print(f"AU extraction error: {e}")
            return {}
    
    def predict_basic(self, face_img):
        """Enhanced emotion prediction using Action Units"""
        try:
            aus = self.extract_action_units(face_img)
            if not aus:
                return "Neutral", 0.5
            self.last_aus = aus

            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)

            emotion_scores = {
                'Happy': 0.0,
                'Sad': 0.0,
                'Angry': 0.0,
                'Fear': 0.0,
                'Surprise': 0.0,
                'Disgust': 0.0,
                'Neutral': 0.6
            }
            # Happy
            happy_score = 0
            if aus.get('AU6_AU12', 0) > 20:
                happy_score += 1.0
            if brightness > 120:
                happy_score += 0.3
            if happy_score > 0.4:
                emotion_scores['Happy'] = happy_score

            # Sad (less sensitive)
            sad_score = 0
            if aus.get('AU15', 0) > 10:  # Increased threshold
                sad_score += 1.0
            if brightness < 105:  # Lowered brightness threshold
                sad_score += 0.2
            if aus.get('AU1_AU2', 0) > 9 and aus.get('AU4', 0) > 9:  # Increased thresholds
                sad_score += 0.2
            if contrast < 25:  # Lowered contrast threshold
                sad_score += 0.1
            if sad_score > 1.0:  # Require more evidence for Sad
                emotion_scores['Sad'] = sad_score

            # Angry
            angry_score = 0
            if aus.get('AU4', 0) > 12:
                angry_score += 1.0
            if contrast > 30:
                angry_score += 0.3
            if aus.get('AU4', 0) > 8 and brightness < 110:
                angry_score += 0.2
            if angry_score > 0.4:
                emotion_scores['Angry'] = angry_score

            # Fear (make more sensitive)
            fear_score = 0
            if aus.get('AU1_AU2', 0) > 8:  # Lowered threshold
                fear_score += 0.7
            if aus.get('AU5', 0) > 6:  # Lowered threshold
                fear_score += 0.5
            if aus.get('AU20', 0) > 4:  # Lowered threshold
                fear_score += 0.3
            if fear_score > 0.4:
                emotion_scores['Fear'] = fear_score

            # Surprise (make more sensitive)
            surprise_score = 0
            # Surprise: AU1+AU2 (brow raiser) + AU25+AU26 (mouth opening/jaw drop)
            surprise_score = 0
            if aus.get('AU1_AU2', 0) > 9:  # Lowered threshold for eyebrow raise
                surprise_score += 0.7
            if aus.get('AU25', 0) > 4 or aus.get('AU26', 0) > 3:  # Lowered thresholds for mouth opening
                surprise_score += 0.7
            # If both are strongly present, boost the score
            if aus.get('AU1_AU2', 0) > 12 and (aus.get('AU25', 0) > 6 or aus.get('AU26', 0) > 5):
                surprise_score += 0.3
            if surprise_score > 0.4:
                emotion_scores['Surprise'] = surprise_score

            # Disgust (make more sensitive)
            disgust_score = 0
            if aus.get('AU9', 0) > 5:  # Lowered threshold
                disgust_score += 0.8
            if aus.get('AU10', 0) > 4:  # Lowered threshold
                disgust_score += 0.6
            if disgust_score > 0.4:
                emotion_scores['Disgust'] = disgust_score

            # Winner-takes-all
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            if max_emotion != 'Neutral' and max_score < 0.7:
                max_emotion = 'Neutral'
                max_score = emotion_scores['Neutral']
            confidence = min(0.95, max_score)
            if confidence > 0.7:
                confidence = min(0.9, confidence + 0.05)
            return max_emotion, confidence

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
                
                # Only process the largest face (if any faces detected)
                if faces:
                    # Find the largest face by area
                    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                    x, y, w, h = largest_face
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