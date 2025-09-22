#!/usr/bin/env python3
"""
Simple Emotion Detection - No External Dependencies
Works without Haar cascade files or complex models
"""

import cv2
import numpy as np
import os
import sys
import glob
import pickle

class SimpleEmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (255, 0, 255),
            'Happy': (0, 255, 255), 'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
            'Surprise': (0, 165, 255)
        }
        
        # Load all available models for ensemble prediction
        self.models = self.load_all_models()
        self.ensemble_weights = self.load_ensemble_weights()
        
        # Keep backward compatibility
        self.model = self.models[0] if self.models else None
        
        # Initialize face detection
        self.face_cascade = self.init_face_detection()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start = cv2.getTickCount()
        self.current_fps = 0
        
        # Debug mode for Action Units and ensemble
        self.debug_mode = False
        self.last_aus = {}
        
        # Print ensemble information
        self.print_ensemble_info()
    
    def load_ensemble_weights(self):
        """Load ensemble weights from pickle file if available"""
        ensemble_info_paths = [
            'models/ensemble_raf_db1_*_info.pkl',
            '../models/ensemble_raf_db1_*_info.pkl'
        ]
        
        for pattern in ensemble_info_paths:
            files = glob.glob(pattern)
            if files:
                try:
                    with open(files[0], 'rb') as f:
                        info = pickle.load(f)
                    if 'weights' in info:
                        print(f"Loaded ensemble weights: {info['weights']}")
                        return info['weights']
                except Exception as e:
                    print(f"Error loading ensemble weights: {e}")
        
        # Default equal weights
        return {}

    def load_all_models(self):
        """Load all available emotion recognition models for ensemble prediction"""
        models = []
        
        # Priority order: finetuned > best > final > simple
        model_patterns = [
            # Finetuned models (highest priority)
            'models/*_finetuned_best.h5',
            '../models/*_finetuned_best.h5',
            
            # Best models
            'models/*_best.h5',
            '../models/*_best.h5',
            
            # Final models
            'models/*_final.h5',
            '../models/*_final.h5',
            
            # Simple/original models
            'models/raf_db_simple_cnn.h5',
            'models/model_file_30epochs.h5',
            '../models/raf_db_simple_cnn.h5',
            '../models/model_file_30epochs.h5'
        ]
        
        loaded_models = set()  # Track loaded model names to avoid duplicates
        
        for pattern in model_patterns:
            model_files = glob.glob(pattern)
            
            for model_path in sorted(model_files, reverse=True):  # Newest first
                # Extract model identifier to avoid loading same architecture twice
                model_name = os.path.basename(model_path).split('_')[0:3]  # e.g., ['raf', 'db1', 'resnet']
                model_id = '_'.join(model_name)
                
                # Skip if we already loaded this model architecture
                if model_id in loaded_models:
                    continue
                
                try:
                    from tensorflow import keras
                    model = keras.models.load_model(model_path)
                    
                    # Validate model architecture
                    if not self.validate_model(model, model_path):
                        print(f"Model validation failed for {model_path}")
                        continue
                    
                    # Determine model type for weighting
                    if "resnet" in model_path.lower():
                        model_type = "resnet"
                    elif "mobilenet" in model_path.lower():
                        model_type = "mobilenet"
                    elif "custom" in model_path.lower():
                        model_type = "custom"
                    else:
                        model_type = "simple"
                    
                    model_info = {
                        'model': model,
                        'path': model_path,
                        'type': model_type,
                        'name': os.path.basename(model_path),
                        'input_shape': model.input_shape,
                        'output_shape': model.output_shape
                    }
                    
                    models.append(model_info)
                    loaded_models.add(model_id)
                    print(f"Loaded {model_type} model: {model_path} (input: {model.input_shape}, output: {model.output_shape})")
                    
                except ImportError:
                    print("TensorFlow not available - skipping neural network models")
                    break
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}")
                    continue
        
        print(f"Total models loaded for ensemble: {len(models)}")
        return models
    
    def validate_model(self, model, model_path):
        """Validate that model has expected input/output structure for emotion recognition"""
        try:
            # Check input shape - should be compatible with (None, 48, 48, 1) or similar
            input_shape = model.input_shape
            if len(input_shape) != 4:  # Should be (batch, height, width, channels)
                print(f"Warning: Unexpected input shape {input_shape} for {model_path}")
                return False
            
            # Check output shape - should have 7 classes for emotions
            output_shape = model.output_shape
            if len(output_shape) != 2 or output_shape[1] != 7:
                print(f"Warning: Unexpected output shape {output_shape} for {model_path} (expected 7 emotions)")
                return False
            
            return True
            
        except Exception as e:
            print(f"Model validation error for {model_path}: {e}")
            return False

    def print_ensemble_info(self):
        """Print information about the loaded ensemble"""
        print("\n" + "="*60)
        print("ENSEMBLE EMOTION DETECTION SYSTEM")
        print("="*60)
        
        if not self.models:
            print("⚠️  No models loaded - using basic detection only")
            return
        
        print(f"✅ Loaded {len(self.models)} models for ensemble prediction:")
        
        for i, model_info in enumerate(self.models, 1):
            model_type = model_info['type']
            weight = self.ensemble_weights.get(model_type, 1.0)
            input_shape = model_info['input_shape']
            
            print(f"   {i}. {model_info['name']}")
            print(f"      Type: {model_type.title()} | Weight: {weight} | Input: {input_shape}")
        
        if self.ensemble_weights:
            print(f"\n🎯 Ensemble weights: {self.ensemble_weights}")
        else:
            print("\n🎯 Using equal weights for all models")
        
        print("="*60 + "\n")
    
    def enable_debug_mode(self):
        """Enable debug mode to see individual model predictions"""
        self.debug_mode = True
        print("🔍 Debug mode enabled - will show individual model predictions")
    
    def disable_debug_mode(self):
        """Disable debug mode"""
        self.debug_mode = False
        print("🔍 Debug mode disabled")
    
    def get_ensemble_summary(self):
        """Get a summary of the ensemble system"""
        return {
            'total_models': len(self.models),
            'model_types': [model['type'] for model in self.models],
            'model_names': [model['name'] for model in self.models],
            'ensemble_weights': self.ensemble_weights,
            'has_ensemble': len(self.models) > 1
        }
    
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
        """Predict emotion with ensemble models or basic rules"""
        if self.models:
            return self.predict_with_ensemble(face_img)
        else:
            return self.predict_basic(face_img)
    
    def predict_with_ensemble(self, face_img):
        """Predict using ensemble of models with weighted voting"""
        try:
            # Get predictions from all models
            all_predictions = []
            model_weights = []
            
            for model_info in self.models:
                try:
                    model = model_info['model']
                    model_type = model_info['type']
                    
                    # Preprocess image specifically for this model
                    preprocessed = self.preprocess_for_specific_model(face_img, model_info)
                    
                    # Get prediction from this model
                    result = model.predict(preprocessed, verbose=0)
                    
                    # Get weight for this model type
                    weight = self.ensemble_weights.get(model_type, 1.0)
                    
                    all_predictions.append(result[0])
                    model_weights.append(weight)
                    
                    if self.debug_mode:
                        print(f"Model {model_info['name']}: {self.emotions[np.argmax(result[0])]} (conf: {np.max(result[0]):.3f}, weight: {weight})")
                    
                except Exception as e:
                    print(f"Error predicting with model {model_info['name']}: {e}")
                    continue
            
            if not all_predictions:
                return self.predict_basic(face_img)
            
            # Combine predictions using weighted average
            ensemble_prediction = self.combine_predictions(all_predictions, model_weights)
            
            # Get final emotion and confidence
            emotion_idx = np.argmax(ensemble_prediction)
            confidence = ensemble_prediction[emotion_idx]
            
            if self.debug_mode:
                print(f"Ensemble result: {self.emotions[emotion_idx]} (conf: {confidence:.3f}) from {len(all_predictions)} models")
            
            return self.emotions[emotion_idx], confidence
            
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return self.predict_basic(face_img)
    
    def preprocess_for_models(self, face_img):
        """Preprocess face image for model input"""
        # Convert to grayscale and resize to 48x48 (standard for emotion models)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        return reshaped
    
    def preprocess_for_specific_model(self, face_img, model_info):
        """Preprocess face image for a specific model's input requirements"""
        try:
            input_shape = model_info['input_shape']
            
            # Handle different input shapes
            if len(input_shape) == 4:  # (batch, height, width, channels)
                height, width, channels = input_shape[1], input_shape[2], input_shape[3]
                
                if channels == 1:  # Grayscale
                    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (width, height))
                    normalized = resized / 255.0
                    reshaped = np.reshape(normalized, (1, height, width, 1))
                elif channels == 3:  # RGB
                    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (width, height))
                    normalized = resized / 255.0
                    reshaped = np.reshape(normalized, (1, height, width, 3))
                else:
                    # Fallback to standard preprocessing
                    return self.preprocess_for_models(face_img)
                
                return reshaped
            else:
                # Fallback to standard preprocessing
                return self.preprocess_for_models(face_img)
                
        except Exception as e:
            print(f"Error preprocessing for model {model_info['name']}: {e}")
            return self.preprocess_for_models(face_img)
    
    def combine_predictions(self, predictions, weights):
        """Combine multiple model predictions using weighted averaging"""
        if len(predictions) == 1:
            return predictions[0]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average of predictions
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, normalized_weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    def predict_with_model(self, face_img):
        """Legacy method - predict using single model (backward compatibility)"""
        if not self.models:
            return self.predict_basic(face_img)
        
        # Use first model for backward compatibility
        try:
            preprocessed = self.preprocess_for_models(face_img)
            result = self.models[0]['model'].predict(preprocessed, verbose=0)
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
            
            # Emotion classification based on Action Units (winner-takes-all, stable)
            emotion_scores = {
                'Happy': 0.0,
                'Sad': 0.0,
                'Angry': 0.0,
                'Fear': 0.0,
                'Surprise': 0.0,
                'Disgust': 0.0,
                'Neutral': 0.6  # Very strong fallback to Neutral
            }
            # Happy: AU6+AU12 (cheek raiser + lip corner puller)
            happy_score = 0
            if aus.get('AU6_AU12', 0) > 20:
                happy_score += 1.0
            if brightness > 120:
                happy_score += 0.3
            if happy_score > 0.4:
                emotion_scores['Happy'] = happy_score
            # Sad: AU15 (lip corner depressor) + additional sad indicators
            sad_score = 0
            if aus.get('AU15', 0) > 6:
                sad_score += 1.0
            if brightness < 120:
                sad_score += 0.3
            if aus.get('AU1_AU2', 0) > 5 and aus.get('AU4', 0) > 5:
                sad_score += 0.2
            if contrast < 30:
                sad_score += 0.1
            if sad_score > 0.4:
                emotion_scores['Sad'] = sad_score
            # Angry: AU4 (brow lowerer) + contrast
            angry_score = 0
            if aus.get('AU4', 0) > 12:
                angry_score += 1.0
            if contrast > 30:
                angry_score += 0.3
            if aus.get('AU4', 0) > 8 and brightness < 110:
                angry_score += 0.2
            if angry_score > 0.4:
                emotion_scores['Angry'] = angry_score
            # Fear: AU1+AU2 (brow raiser) + AU5 (upper lid raiser) + AU20 (lip stretcher)
            fear_score = 0
            if aus.get('AU1_AU2', 0) > 10:
                fear_score += 0.7
            if aus.get('AU5', 0) > 8:
                fear_score += 0.5
            if aus.get('AU20', 0) > 6:
                fear_score += 0.3
            if fear_score > 0.4:
                emotion_scores['Fear'] = fear_score
            # Surprise: AU1+AU2 (brow raiser) + AU5 (upper lid raiser) + AU25+AU26 (jaw drop)
            surprise_score = 0
            if aus.get('AU1_AU2', 0) > 14:
                surprise_score += 0.7
            if aus.get('AU5', 0) > 10:
                surprise_score += 0.5
            if aus.get('AU25', 0) > 6 or aus.get('AU26', 0) > 5:
                surprise_score += 0.5
            if surprise_score > 0.4:
                emotion_scores['Surprise'] = surprise_score
            # Disgust: AU9 (nose wrinkler) + AU10 (upper lip raiser)
            disgust_score = 0
            if aus.get('AU9', 0) > 7:
                disgust_score += 0.8
            if aus.get('AU10', 0) > 6:
                disgust_score += 0.6
            if disgust_score > 0.4:
                emotion_scores['Disgust'] = disgust_score
            # Winner-takes-all: only allow non-neutral if it is clearly dominant
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