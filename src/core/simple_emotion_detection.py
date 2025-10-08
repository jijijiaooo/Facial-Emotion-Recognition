#!/usr/bin/env python3
"""
Simple Emotion Detection - No External Dependenci            'Disgust': {
                'primary': [('AU9', 30), ('AU10', 12)],  # Nose wrinkler and upper lip raiser
                'secondary': [('AU4', 20), ('AU7', 10), ('AU16', 8), ('AU24', 12)],  # Brow lowerer, lid tightener, lower lip depressor, lip pressor
                'inhibitors': [('AU12', 15), ('AU26', 25)]  # Block lip corner pulling and strong jaw drop
            }rks without Haar cascade files or complex models
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
        
        # Temporal smoothing for stable detection
        self.emotion_history = []
        self.confidence_history = []
        self.history_size = 3  # Reduced from 5 to 3 for more responsiveness
        self.gui_mode = False  # Flag for GUI mode with more responsive detection
        
        # Adaptive thresholds (will adjust based on face characteristics)
        self.adaptive_thresholds = {
            'brightness_baseline': 120,
            'contrast_baseline': 30,
            'face_size_factor': 1.0
        }
        
        # Advanced AU combination patterns
        self.emotion_patterns = self._init_emotion_patterns()

    def _init_emotion_patterns(self):
        """Initialize enhanced emotion detection patterns with improved AU discrimination"""
        # Enhanced patterns to address specific confusion issues:
        # - Surprise vs Happy (mouth opening context)
        # - Fear vs Surprise (eye widening distinction) 
        # - Sad vs Angry (furrowing intensity)
        return {
            'Happy': {
                'primary': [('AU12', 4)],  # Lip corner puller - reasonable threshold
                'secondary': [('AU6', 6), ('AU7', 8), ('AU14', 5)],  # Normal thresholds
                'inhibitors': [('AU4', 8), ('AU15', 8), ('AU16', 10)]  # Reasonable inhibitors
            },
            'Sad': {
                'primary': [('AU15', 4), ('AU17', 5)],  # Balanced thresholds for sadness - no brow bias
                'secondary': [('AU1', 3), ('AU4', 6), ('AU11', 5), ('AU16', 6)],  # Move AU4 to secondary to reduce bias
                'inhibitors': [('AU12', 6), ('AU6', 8)]  # Keep happiness inhibitors
            },
            'Angry': {
                'primary': [('AU4', 6)],  # Balanced brow furrow threshold for anger
                'secondary': [('AU9', 12), ('AU7', 6)],  # Nose wrinkle, lid tightener - balanced thresholds
                'inhibitors': [('AU12', 8), ('AU6', 10)]  # Simple happiness inhibitors
            },
            'Fear': {
                'primary': [('AU5', 12)],  # Eyes wide - PRIMARY fear indicator (lowered threshold for better detection)
                'secondary': [('AU1', 4), ('AU2', 4), ('AU7', 10), ('AU20', 6)],  # Brow raise, lid tightener, lip stretcher (optional)
                'inhibitors': [('AU12', 12), ('AU26', 20)]  # Block strong smile and full jaw drop (that's surprise)
            },
            'Surprise': {
                'primary': [('AU1', 3), ('AU2', 5), ('AU26', 15)],  # Brow raise AND jaw drop required - mouth must be fully opened
                'secondary': [('AU5', 10), ('AU25', 12)],  # Eye widening and lips parting are bonuses
                'inhibitors': []  # No inhibitors for surprise
            },
            'Disgust': {
                'primary': [('AU9', 50)],  # Much lower nose crunch threshold - more realistic
                'secondary': [('AU4', 25), ('AU10', 15), ('AU38', 30)],  # Lower thresholds for all indicators
                'inhibitors': [('AU6_AU12', 15), ('AU26', 40)]  # Reduced inhibitors, removed brow raise inhibitor
            }
        }

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
                    input_shape = model.input_shape
                    model_info = {
                        'type': 'keras',
                        'model': model,
                        'input_shape': input_shape,
                        'path': model_path
                    }
                    loaded_models.append(model_info)
                    print(f"Loaded Keras model: {model_path} (input shape: {input_shape})")
                except Exception as e:
                    print(f"Failed to load Keras model {model_path}: {e}")
            elif model_path.endswith('.pkl') and os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    model_info = {
                        'type': 'sklearn',
                        'model': model,
                        'input_shape': None,  # sklearn models don't have fixed input shapes
                        'path': model_path
                    }
                    loaded_models.append(model_info)
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
    
    def preprocess_face_48x48(self, face_img):
        """Resize and normalize face image for 48x48x1 models (e.g., simple CNN)."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        return reshaped
    
    def preprocess_face_224x224(self, face_img):
        """Resize and normalize face image for 224x224x3 models (e.g., ResNet, MobileNet)."""
        # Convert to RGB if needed (OpenCV uses BGR)
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        return reshaped

    def predict_emotion(self, face_img):
        """Predict emotion using ensemble of models or fallback to basic rules."""
        if self.models:
            if self.debug_mode:
                print(f"Using ML models for prediction ({len(self.models)} models loaded)")
            votes = []
            probs = []
            for model_info in self.models:
                try:
                    model_type = model_info['type']
                    model = model_info['model']
                    input_shape = model_info['input_shape']
                    
                    if model_type == 'keras':
                        # Use appropriate preprocessing based on input shape
                        if input_shape and len(input_shape) >= 3:
                            height, width = input_shape[1], input_shape[2]
                            if height == 224 and width == 224:
                                img = self.preprocess_face_224x224(face_img)
                            elif height == 48 and width == 48:
                                img = self.preprocess_face_48x48(face_img)
                            else:
                                print(f"Unsupported input shape: {input_shape}")
                                continue
                        else:
                            # Default to 224x224 for unknown shapes
                            img = self.preprocess_face_224x224(face_img)
                        
                        pred = model.predict(img, verbose=0)
                        idx = int(np.argmax(pred))
                        votes.append(idx)
                        probs.append(pred[0])
                        
                    elif model_type == 'sklearn':
                        # For sklearn models, use 48x48 preprocessing and flatten
                        img = self.preprocess_face_48x48(face_img)
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
                    print(f"Model prediction error ({model_info.get('path', 'unknown')}): {e}")
            if votes:
                # Use weighted voting based on confidence instead of simple majority
                emotion_weights = [0.0] * len(self.emotions)
                
                # Check for systematic bias - count votes for each emotion
                vote_counts = {}
                for vote in votes:
                    vote_counts[vote] = vote_counts.get(vote, 0) + 1
                
                # Detect if Surprise is being over-voted (>=25% of models, was >50%)
                surprise_overvote = vote_counts.get(6, 0) >= len(votes) * 0.25
                surprise_heavy_overvote = vote_counts.get(6, 0) >= len(votes) * 0.6  # 60%+ Surprise votes
                
                # Weighted voting with enhanced bias correction
                for vote, prob in zip(votes, probs):
                    weight = prob[vote]  # Use confidence as weight
                    
                    # Apply aggressive bias corrections for common overdetections
                    if vote == 6:  # Surprise - apply correction to ALL Surprise votes
                        if surprise_overvote:  # If >=25% of models vote Surprise
                            weight *= 0.05  # Reduce by 95% when overvoted
                        else:
                            weight *= 0.3  # Reduce Surprise by 70% (less aggressive)
                    elif vote == 3:  # Happy - check for over-detection 
                        happy_count = vote_counts.get(3, 0)
                        neutral_count = vote_counts.get(4, 0)
                        
                        # Special case: If competing with Neutral, be more aggressive
                        if neutral_count > 0 and happy_count > neutral_count:
                            weight *= 0.25  # Heavily reduce Happy when competing with Neutral
                        elif happy_count >= len(votes) * 0.3:  # If >=30% vote Happy
                            weight *= 0.4  # Reduce Happy by 60% when it's overrepresented
                        elif weight < 0.5:  # Reduce weak Happy predictions
                            weight *= 0.6  # Reduce by 40%
                    elif vote == 4:  # Neutral - boost when competing with Happy or Sad
                        happy_count = vote_counts.get(3, 0)
                        sad_count = vote_counts.get(5, 0)
                        neutral_count = vote_counts.get(4, 0)
                        
                        # Boost Neutral when it's close to Happy in vote count
                        if happy_count > 0 and neutral_count > 0:
                            if happy_count > neutral_count and (happy_count - neutral_count) <= 3:
                                weight *= 1.5  # Boost Neutral by 50%
                        
                        # Also boost Neutral when competing with Sad
                        if sad_count > 0 and neutral_count > 0:
                            if sad_count > neutral_count and (sad_count - neutral_count) <= 3:
                                weight *= 1.4  # Boost Neutral by 40% when competing with Sad
                        
                        # Special case: When Surprise is heavily suppressed (60%+), boost Neutral
                        if surprise_heavy_overvote:
                            weight *= 2.0  # Strong boost to Neutral when Surprise is over-detected
                                
                    elif vote == 5:  # Sad - No special treatment
                        # Treat sadness like any other emotion - no boosts or penalties
                        pass
                    
                    emotion_weights[vote] += weight
                
                # Find the emotion with highest weighted score
                final_idx = int(np.argmax(emotion_weights))
                max_weight = emotion_weights[final_idx]
                
                # Fallback mechanism: If Surprise is heavily over-detected (60%+) 
                # and winning emotion has very low confidence, default to Neutral
                if surprise_heavy_overvote and max_weight < 1.0:
                    if final_idx in [5, 1, 2]:  # If winner is Sad, Disgust, or Fear (negative emotions)
                        if self.debug_mode:
                            print(f"  Fallback: Surprise heavily over-detected ({vote_counts.get(6, 0)}/{len(votes)} votes), low confidence winner {self.emotions[final_idx]} ({max_weight:.2f}), defaulting to Neutral")
                        final_idx = 4  # Force Neutral
                        confidence = 0.3  # Moderate confidence for fallback
                    else:
                        # Calculate normal confidence
                        winning_probs = [prob[final_idx] for prob in probs if len(prob) > final_idx and prob[final_idx] > 0.1]
                        confidence = float(np.mean(winning_probs)) if winning_probs else max_weight
                else:
                    # Calculate normal confidence
                    winning_probs = [prob[final_idx] for prob in probs if len(prob) > final_idx and prob[final_idx] > 0.1]
                    confidence = float(np.mean(winning_probs)) if winning_probs else max_weight
                
                emotion = self.emotions[final_idx]
                
                # No AU-based overrides - use ML models as primary system
                if self.debug_mode:
                    try:
                        au_emotion, au_confidence = self.predict_basic(face_img)
                        print(f"AU Info: AU detects {au_emotion} ({au_confidence:.3f}), ML detects {emotion} ({confidence:.3f}) - using ML result")
                    except Exception as e:
                        print(f"AU debug error: {e}")
                
                if self.debug_mode:
                    print(f"ML Prediction: {emotion} (confidence: {confidence:.3f}, weights: {[f'{self.emotions[i]}={w:.2f}' for i, w in enumerate(emotion_weights) if w > 0.1]})")
                    print(f"  Original votes: {votes}")
                
                return emotion, confidence
        # Fallback
        if self.debug_mode:
            print("Falling back to basic emotion detection (no ML votes)")
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

            # Enhanced AU1/AU2: Brow raiser with context-aware detection
            brow_left_mean = np.mean(regions['eyebrow_left'])
            brow_right_mean = np.mean(regions['eyebrow_right'])
            forehead_mean = np.mean(regions['forehead'])
            brow_raise = ((brow_left_mean + brow_right_mean) / 2) - forehead_mean
            brow_std = (np.std(regions['eyebrow_left']) + np.std(regions['eyebrow_right'])) / 2
            
            # Calculate different types of brow raising
            # Base brow raise measurement
            base_brow_raise = brow_std + brow_raise / 4
            
            # Get mouth and eye context for distinguishing emotions
            mouth_vertical = np.max(np.mean(regions['mouth'], axis=1)) - np.min(np.mean(regions['mouth'], axis=1))
            eye_area_left = np.std(regions['eye_left']) * np.mean(regions['eye_left'])
            eye_area_right = np.std(regions['eye_right']) * np.mean(regions['eye_right'])
            eye_widening = (eye_area_left + eye_area_right) / 2
            
            # Separate AU1 (Inner Brow Raiser) and AU2 (Outer Brow Raiser)
            brow_h, brow_w = regions['eyebrow_left'].shape
            inner_brow_left = regions['eyebrow_left'][:, :brow_w//3] if brow_w > 3 else regions['eyebrow_left']
            outer_brow_left = regions['eyebrow_left'][:, -brow_w//3:] if brow_w > 3 else regions['eyebrow_left']
            inner_brow_right = regions['eyebrow_right'][:, -brow_w//3:] if brow_w > 3 else regions['eyebrow_right']
            outer_brow_right = regions['eyebrow_right'][:, :brow_w//3] if brow_w > 3 else regions['eyebrow_right']
            
            # AU1: Inner brow raiser (medial portion)
            inner_raise_left = forehead_mean - np.mean(inner_brow_left) + np.std(inner_brow_left)
            inner_raise_right = forehead_mean - np.mean(inner_brow_right) + np.std(inner_brow_right)
            aus['AU1'] = (inner_raise_left + inner_raise_right) / 2
            
            # AU2: Outer brow raiser (lateral portion)  
            outer_raise_left = forehead_mean - np.mean(outer_brow_left) + np.std(outer_brow_left)
            outer_raise_right = forehead_mean - np.mean(outer_brow_right) + np.std(outer_brow_right)
            aus['AU2'] = (outer_raise_left + outer_raise_right) / 2
            
            # Combined AU1_AU2 for backward compatibility
            aus['AU1_AU2'] = (aus['AU1'] + aus['AU2']) / 2

            # Enhanced AU4: Brow lowerer with intensity distinction
            brow_lower_base = 255 - ((brow_left_mean + brow_right_mean) / 2)
            
            # Calculate furrowing intensity
            brow_gradient_left = np.abs(np.diff(regions['eyebrow_left'])).sum()
            brow_gradient_right = np.abs(np.diff(regions['eyebrow_right'])).sum()
            furrowing_intensity = (brow_gradient_left + brow_gradient_right) / 2
            
            # Distinguish sad (gentle) vs angry (intense) furrowing  
            if furrowing_intensity > 50:  # Intense furrowing (Angry)
                aus['AU4'] = brow_lower_base + brow_std / 2 + furrowing_intensity / 5
            else:  # Gentle furrowing (Sad) - don't reduce the score
                aus['AU4'] = brow_lower_base + brow_std / 3
                
            # Add looking down detection for sadness (AU61 equivalent)
            # Check if eyebrow area is darker than usual (looking down)
            if np.mean(regions['eyebrow_left']) < forehead_mean - 10 or np.mean(regions['eyebrow_right']) < forehead_mean - 10:
                aus['AU4'] += 20  # Boost AU4 for looking down gesture

            # Enhanced AU5: Eye widening with precise measurement
            eye_height_left = np.max(regions['eye_left']) - np.min(regions['eye_left'])
            eye_height_right = np.max(regions['eye_right']) - np.min(regions['eye_right'])
            eye_openness = (np.std(regions['eye_left']) + np.std(regions['eye_right'])) / 2
            eye_brightness = (np.mean(regions['eye_left']) + np.mean(regions['eye_right'])) / 2
            
            # Calculate actual eye widening (for Fear detection)
            eye_height_avg = (eye_height_left + eye_height_right) / 2
            eye_widening_score = eye_openness + eye_brightness / 10 + eye_height_avg / 15
            aus['AU5'] = eye_widening_score
            
            # New AU for precise Fear detection
            aus['AU5_wide'] = eye_widening  # Direct eye widening measurement

            # AU6: Cheek raiser (orbicularis oculi) - eye region changes during smile
            eye_squeeze_left = np.std(regions['eye_left']) + (255 - np.mean(regions['eye_left'])) / 10
            eye_squeeze_right = np.std(regions['eye_right']) + (255 - np.mean(regions['eye_right'])) / 10  
            aus['AU6'] = (eye_squeeze_left + eye_squeeze_right) / 2
            
            # AU12: Lip corner puller (zygomatic major) - lateral mouth movement
            mouth_h, mouth_w = regions['mouth'].shape
            if mouth_w > 4:
                left_mouth_corner = regions['mouth'][:, :mouth_w//4]
                right_mouth_corner = regions['mouth'][:, -mouth_w//4:]
                mouth_center = regions['mouth'][:, mouth_w//3:2*mouth_w//3]
                
                # Lip corner pulling creates brightness difference
                corner_pull = (np.mean(left_mouth_corner) + np.mean(right_mouth_corner))/2 - np.mean(mouth_center)
                aus['AU12'] = max(0, corner_pull + np.std(regions['mouth'], axis=0).mean())
            else:
                aus['AU12'] = np.std(regions['mouth'])
                
            # Combined AU6_AU12 for backward compatibility
            aus['AU6_AU12'] = (aus['AU6'] + aus['AU12']) / 2
            
            # AU7: Lid tightener - tension around eyes
            eye_tension_left = np.var(regions['eye_left']) + np.std(regions['eye_left'])
            eye_tension_right = np.var(regions['eye_right']) + np.std(regions['eye_right'])
            aus['AU7'] = (eye_tension_left + eye_tension_right) / 2

            # AU9: Nose wrinkler/crunch (enhanced detection)
            nose_std = np.std(regions['nose'])
            nose_contrast = np.max(regions['nose']) - np.min(regions['nose'])
            
            # Detect wrinkle patterns in nose region
            if regions['nose'].size > 0:
                # Horizontal wrinkle detection (typical for nose crunch)
                horizontal_grad = np.abs(np.diff(regions['nose'], axis=0))
                horizontal_wrinkles = np.mean(horizontal_grad)
                
                # Vertical compression (nose narrowing during crunch)
                vertical_grad = np.abs(np.diff(regions['nose'], axis=1))
                vertical_compression = np.mean(vertical_grad)
                
                # Combined nose crunch score
                nose_crunch = nose_std + nose_contrast/8 + horizontal_wrinkles*2 + vertical_compression
            else:
                nose_crunch = nose_std
                
            aus['AU9'] = nose_crunch
            
            # AU38: Nostril dilator (additional nose AU for more detailed detection)
            # Detect nostril flaring/dilation patterns
            if regions['nose'].size > 4:  # Ensure we have enough data
                nose_width_var = np.var(np.mean(regions['nose'], axis=0))
                nostril_activity = nose_width_var + np.std(regions['nose'][:, :regions['nose'].shape[1]//3])
                aus['AU38'] = nostril_activity
            else:
                aus['AU38'] = 0

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
            
            # AU11: Nasolabial deepener - deepening of nasolabial fold
            if mouth_h > 2 and mouth_w > 4:
                nose_to_mouth_area = regions['mouth'][:mouth_h//2, :]  # Upper mouth area
                nasolabial_depth = np.std(nose_to_mouth_area) + (255 - np.mean(nose_to_mouth_area)) / 8
                aus['AU11'] = nasolabial_depth
            else:
                aus['AU11'] = 0
                
            # AU13: Cheek puffer - cheek expansion
            # Approximated by analyzing mouth width vs typical proportions
            mouth_width_expansion = np.std(regions['mouth'], axis=1).mean()
            aus['AU13'] = mouth_width_expansion
            
            # AU14: Dimpler - dimple formation (corner mouth depression)
            if mouth_w > 6:
                left_dimple_area = regions['mouth'][:, mouth_w//6:mouth_w//3]
                right_dimple_area = regions['mouth'][:, -mouth_w//3:-mouth_w//6]
                dimple_depression = (np.mean(left_dimple_area) + np.mean(right_dimple_area))/2
                aus['AU14'] = max(0, 255 - dimple_depression)
            else:
                aus['AU14'] = 0
                
            # AU16: Lower lip depressor - lower lip pulling down
            if mouth_h > 2:
                lower_lip_area = regions['mouth'][mouth_h//2:, :]
                lower_lip_depression = (255 - np.mean(lower_lip_area)) + np.std(lower_lip_area)
                aus['AU16'] = lower_lip_depression
            else:
                aus['AU16'] = np.std(regions['mouth_lower'])
                
            # Enhanced AU17: Chin raiser - critical for pouting detection
            jaw_lower_area = regions['jaw'][regions['jaw'].shape[0]//2:, :] if regions['jaw'].size > 0 else regions['jaw']
            chin_tension = np.std(jaw_lower_area) * 3  # Increased sensitivity
            chin_darkness = (255 - np.mean(jaw_lower_area))
            mental_muscle_activity = chin_tension + chin_darkness / 5
            aus['AU17'] = mental_muscle_activity
            
            # Enhanced AU18: Lip puckerer - enhanced for pouting
            # Pouting creates lip protrusion and rounding
            if mouth_h > 2 and mouth_w > 4:
                mouth_center = regions['mouth'][mouth_h//3:2*mouth_h//3, mouth_w//3:2*mouth_w//3]
                lip_concentration = np.std(mouth_center) * 2
                lip_protrusion_score = (255 - np.mean(mouth_center)) / 3
                vertical_compression = max(0, 15 - mouth_vertical)  # Less vertical opening = more puckering
                aus['AU18'] = lip_concentration + lip_protrusion_score + vertical_compression
            else:
                mouth_roundness = np.var(regions['mouth']) / (np.mean(regions['mouth']) + 1e-5)
                lip_protrusion = np.std(regions['mouth_lower']) + np.std(regions['mouth_upper'])
                aus['AU18'] = mouth_roundness * 10 + lip_protrusion
            
            # Enhanced AU22: Lip funneler - critical for pouting (lips narrow and protrude)
            if mouth_w > 4:
                mouth_center_narrow = regions['mouth'][:, mouth_w//3:2*mouth_w//3]
                mouth_edges = np.concatenate([regions['mouth'][:, :mouth_w//4], regions['mouth'][:, -mouth_w//4:]], axis=1)
                
                # Funneling creates contrast between center (concentrated) and edges
                center_concentration = np.std(mouth_center_narrow) * 3
                edge_vs_center = np.mean(mouth_edges) - np.mean(mouth_center_narrow)
                lip_narrowing = max(0, edge_vs_center / 2)
                aus['AU22'] = center_concentration + lip_narrowing
            else:
                aus['AU22'] = np.std(regions['mouth']) * 2
                
            # Enhanced AU23: Lip tightener - tension in lip muscles during pouting
            upper_lip_tension = np.var(regions['mouth_upper']) + np.std(regions['mouth_upper']) * 2
            lower_lip_tension = np.var(regions['mouth_lower']) + np.std(regions['mouth_lower']) * 2
            lip_contact_tension = abs(np.mean(regions['mouth_upper']) - np.mean(regions['mouth_lower'])) * 2
            aus['AU23'] = (upper_lip_tension + lower_lip_tension) / 2 + lip_contact_tension
            
            # Enhanced AU24: Lip pressor - lips pressed together (strong in pouting)
            lip_contact_pressure = abs(np.mean(regions['mouth_upper']) - np.mean(regions['mouth_lower'])) * 3
            vertical_compression = max(0, 20 - mouth_vertical)  # Strong reward for closed mouth
            horizontal_compression = np.std(regions['mouth'], axis=1).mean() * 2
            aus['AU24'] = lip_contact_pressure + vertical_compression + horizontal_compression
            
            # AU27: Mouth stretch - horizontal mouth opening (like during scream)
            if mouth_w > 4:
                mouth_stretch = mouth_vertical * 2 + np.std(regions['mouth'], axis=0).mean() * 3
                aus['AU27'] = mouth_stretch
            else:
                aus['AU27'] = mouth_vertical
                
            # AU28: Lip suck - lips drawn inward
            lip_inward_pull = (np.mean(regions['mouth_upper']) + np.mean(regions['mouth_lower']))/2
            background_brightness = (forehead_mean + np.mean(regions['jaw'])) / 2
            lip_suck_indicator = max(0, background_brightness - lip_inward_pull)
            aus['AU28'] = lip_suck_indicator + (255 - mouth_vertical * 10)  # Reduced opening suggests sucking
            
            # AU61-like: Downward gaze detection for sadness
            # Compare upper vs lower eye region brightness
            eye_upper_left = regions['eye_left'][:regions['eye_left'].shape[0]//2, :]
            eye_lower_left = regions['eye_left'][regions['eye_left'].shape[0]//2:, :]
            eye_upper_right = regions['eye_right'][:regions['eye_right'].shape[0]//2, :]
            eye_lower_right = regions['eye_right'][regions['eye_right'].shape[0]//2:, :]
            
            if eye_upper_left.size > 0 and eye_lower_left.size > 0:
                gaze_score_left = np.mean(eye_upper_left) - np.mean(eye_lower_left)
                gaze_score_right = np.mean(eye_upper_right) - np.mean(eye_lower_right)
                downward_gaze = (gaze_score_left + gaze_score_right) / 2
                aus['AU_gaze_down'] = max(0, downward_gaze)
            else:
                aus['AU_gaze_down'] = 0

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
    
    def _update_adaptive_thresholds(self, face_img):
        """Update adaptive thresholds based on face characteristics"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Adjust thresholds based on face size
        face_area = h * w
        if face_area > 5000:  # Large face
            self.adaptive_thresholds['face_size_factor'] = 1.2
        elif face_area < 2500:  # Small face
            self.adaptive_thresholds['face_size_factor'] = 0.8
        else:
            self.adaptive_thresholds['face_size_factor'] = 1.0
        
        # Update baseline brightness and contrast
        current_brightness = np.mean(gray)
        current_contrast = np.std(gray)
        
        # Smoothly adapt baselines
        self.adaptive_thresholds['brightness_baseline'] = (
            0.9 * self.adaptive_thresholds['brightness_baseline'] + 
            0.1 * current_brightness
        )
        self.adaptive_thresholds['contrast_baseline'] = (
            0.9 * self.adaptive_thresholds['contrast_baseline'] + 
            0.1 * current_contrast
        )
    
    def _temporal_smoothing(self, emotion, confidence):
        """Apply temporal smoothing to reduce detection jitter"""
        # Add to history
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        
        # Keep history size manageable
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
        
        # If we don't have enough history, return current
        # In GUI mode, be more responsive (require less history)
        min_history = 2 if self.gui_mode else 3
        if len(self.emotion_history) < min_history:
            return emotion, confidence
        
        # Count emotion occurrences in recent history
        emotion_counts = {}
        for hist_emotion in self.emotion_history:
            emotion_counts[hist_emotion] = emotion_counts.get(hist_emotion, 0) + 1
        
        # Find most common emotion
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        most_common_emotion, count = most_common
        
        # Use more responsive smoothing - allow single strong predictions to show through
        # In GUI mode, be even more responsive
        if self.gui_mode:
            # GUI mode: accept single occurrence if high confidence, or 2+ occurrences
            if count >= 1 and (confidence > 0.4 or count >= 2):
                # Use average confidence for that emotion
                relevant_confidences = [
                    conf for em, conf in zip(self.emotion_history, self.confidence_history)
                    if em == most_common_emotion
                ]
                if relevant_confidences:
                    avg_confidence = np.mean(relevant_confidences)
                    return most_common_emotion, avg_confidence
            else:
                return emotion, confidence  # Use current emotion directly
        else:
            # Non-GUI mode: original logic
            if count >= 2 or (count == 1 and confidence > 0.6):  # 2+ occurrences OR 1 high-confidence
                # Use average confidence for that emotion
                relevant_confidences = [
                    conf for em, conf in zip(self.emotion_history, self.confidence_history)
                    if em == most_common_emotion
                ]
                if relevant_confidences:
                    avg_confidence = np.mean(relevant_confidences)
                    return most_common_emotion, avg_confidence
        
        # Otherwise, fall back to current detection
        return emotion, confidence
    
    def _advanced_emotion_scoring(self, aus, brightness, contrast):
        """Advanced emotion scoring using pattern matching"""
        emotion_scores = {'Neutral': 0.5}  # Balanced neutral baseline
        
        # Debug output for emotion detection - focus on pouting/sadness
        if self.gui_mode and hasattr(self, 'debug_counter'):
            self.debug_counter += 1
            if self.debug_counter % 5 == 0:  # Print frequently for pouting debugging
                sad_score = emotion_scores.get('Sad', 0)
                print(f"POUT Debug - AU15(corner↓): {aus.get('AU15', 0):.1f}, AU17(chin): {aus.get('AU17', 0):.1f}, AU18(pucker): {aus.get('AU18', 0):.1f}, AU23(tight): {aus.get('AU23', 0):.1f}, AU24(press): {aus.get('AU24', 0):.1f} | Sad Score: {sad_score:.2f}")
        elif self.gui_mode:
            self.debug_counter = 1
        
        # Adjust thresholds based on adaptive factors
        size_factor = self.adaptive_thresholds['face_size_factor']
        brightness_baseline = self.adaptive_thresholds['brightness_baseline']
        contrast_baseline = self.adaptive_thresholds['contrast_baseline']
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0.0
            
            # Check primary requirements (must be met)
            primary_met = True
            for au_name, threshold in patterns['primary']:
                if au_name == 'brightness':
                    value = brightness
                elif au_name == 'contrast':
                    value = contrast
                else:
                    value = aus.get(au_name, 0)
                
                adjusted_threshold = threshold * size_factor
                if value < adjusted_threshold:
                    primary_met = False
                    break
                else:
                    # Add score for meeting primary requirement
                    score += min(1.5, value / adjusted_threshold)
            
            if not primary_met:
                continue
            
            # Check secondary indicators (bonus points)
            for au_name, threshold in patterns.get('secondary', []):
                if au_name == 'brightness':
                    value = brightness
                    baseline = brightness_baseline
                elif au_name == 'contrast':
                    value = contrast  
                    baseline = contrast_baseline
                else:
                    value = aus.get(au_name, 0)
                    baseline = threshold
                
                adjusted_threshold = threshold * size_factor
                if value > adjusted_threshold:
                    # Give extra bonuses to sad and surprise to help them compete - especially sad
                    if emotion == 'Sad':
                        bonus_multiplier = 1.0  # Double bonus for sadness
                    elif emotion == 'Surprise':
                        bonus_multiplier = 0.6
                    else:
                        bonus_multiplier = 0.3
                    score += bonus_multiplier * (value / adjusted_threshold)
            
            # Check inhibitors (reduce score if present)
            for au_name, threshold in patterns.get('inhibitors', []):
                value = aus.get(au_name, 0)
                adjusted_threshold = threshold * size_factor
                if value > adjusted_threshold:
                    score *= 0.8  # Reduced penalty to allow more emotion detection
            
            # Only add to emotion scores if score is substantial - equal thresholds
            min_threshold = 0.3  # Same threshold for all emotions
            if score > min_threshold:
                emotion_scores[emotion] = score
        
        # No special combination bonuses - treat all emotions equally
        
        return emotion_scores

    def predict_basic(self, face_img):
        """Advanced emotion prediction using enhanced Action Units and temporal smoothing"""
        try:
            # Extract Action Units
            aus = self.extract_action_units(face_img)
            if not aus:
                return "Neutral", 0.5
            
            # Store for debug display
            self.last_aus = aus

            # Update adaptive thresholds based on current face
            self._update_adaptive_thresholds(face_img)

            # Convert to grayscale for additional features
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)

            # Use advanced emotion scoring system
            emotion_scores = self._advanced_emotion_scoring(aus, brightness, contrast)

            # Enhanced winner-takes-all with multi-level validation
            max_emotion = max(emotion_scores, key=emotion_scores.get)
            max_score = emotion_scores[max_emotion]
            
            # Level 1: Emotions must reasonably beat neutral to be detected
            if max_emotion != 'Neutral':
                neutral_score = emotion_scores.get('Neutral', 0.5)
                threshold_bonus = 0.2  # Same threshold for all emotions - no bias
                if max_score < neutral_score + threshold_bonus:
                    max_emotion = 'Neutral'
                    max_score = neutral_score
            
            # Level 2: Same minimum scores for all emotions - no bias
            min_required = 0.4  # Same threshold for all emotions
            if max_emotion != 'Neutral' and max_score < min_required:
                max_emotion = 'Neutral'
                max_score = emotion_scores.get('Neutral', 0.5)
            
            # Level 3: Apply temporal smoothing for stability
            smoothed_emotion, smoothed_confidence = self._temporal_smoothing(max_emotion, max_score)
            
            # Final confidence calibration
            final_confidence = min(0.95, smoothed_confidence * 0.9)  # More conservative confidence
            if final_confidence > 0.8:
                final_confidence = min(0.9, final_confidence + 0.05)
                
            return smoothed_emotion, final_confidence

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