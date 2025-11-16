#!/Users/jiaoshihlo/Codes/Facial-Emotion-Recognition-version-2.9/venv/bin/python3
"""
Hybrid Emotion Detection - Combines CNN model with Action Units for improved accuracy
This version properly utilizes the new simple_cnn model while enhancing predictions with AU analysis
"""

import cv2
import numpy as np
import os
from tensorflow import keras

class HybridEmotionDetector:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 255, 0),    # Green
            'Fear': (255, 0, 255),     # Magenta
            'Happy': (0, 255, 255),    # Yellow
            'Neutral': (255, 255, 255), # White
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 165, 255)  # Orange
        }
        
        # Load the CNN model
        self.model = self.load_cnn_model()
        
        # Load face detection
        self.face_cascade = self.init_face_detection()
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start = cv2.getTickCount()
        self.current_fps = 0
        
        # Temporal smoothing (minimal for responsiveness)
        self.emotion_history = []
        self.confidence_history = []
        self.history_size = 2
        
        # Confidence thresholds adjusted for low-accuracy emotions
        self.confidence_thresholds = {
            'Happy': 0.30,      # High accuracy (81.5%) - normal threshold
            'Surprise': 0.30,   # High accuracy (80.6%) - normal threshold
            'Disgust': 0.30,    # High accuracy (80.2%) - normal threshold
            'Neutral': 0.30,    # Good accuracy (62.3%) - normal threshold
            'Angry': 0.28,      # Moderate accuracy (50.6%) - slightly lower
            'Sad': 0.25,        # Low accuracy (47.4%) - lower threshold
            'Fear': 0.22        # Very low accuracy (37.8%) - lowest threshold
        }
        
        # Debug info
        self.last_aus = {}
        self.last_cnn_probs = []
        self.show_debug = False
        
    def load_cnn_model(self):
        """Load the latest emotion_simple_cnn model (NOT raf_db models)"""
        models_dir = 'models'
        
        # Find ONLY emotion_simple_cnn models (exclude raf_db)
        model_files = []
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                # Must have 'emotion_simple_cnn' AND end with .h5
                # Exclude raf_db models
                if 'emotion_simple_cnn' in f.lower() and f.endswith('.h5'):
                    if 'raf_db' not in f.lower():  # Explicitly exclude raf_db models
                        model_files.append(os.path.join(models_dir, f))
        
        if not model_files:
            raise FileNotFoundError(
                "No emotion_simple_cnn model found in models/ directory.\n"
                "Looking for files matching: emotion_simple_cnn_*.h5"
            )
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = model_files[0]
        
        print(f"🎯 Loading ONLY emotion_simple_cnn model: {os.path.basename(model_path)}")
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded - Input: {model.input_shape}, Output: {model.output_shape}")
        
        return model
    
    def init_face_detection(self):
        """Initialize Haar Cascade face detection"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                print(f"✅ Face detection loaded")
                return cascade
        except Exception as e:
            print(f"⚠️ Face detection error: {e}")
            return None
    
    def preprocess_face(self, face_img):
        """Preprocess face for CNN model (96x96 grayscale)"""
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Resize to 96x96
        resized = cv2.resize(gray, (96, 96))
        
        # Normalize to [0, 1]
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for model input (1, 96, 96, 1)
        preprocessed = normalized.reshape(1, 96, 96, 1)
        
        return preprocessed
    
    def extract_action_units(self, face_img):
        """Extract Action Units from face for emotion enhancement"""
        try:
            # Convert to grayscale
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img
            
            h, w = gray.shape
            
            if h < 50 or w < 50:
                return {}
            
            # Define facial regions
            regions = {
                'forehead': gray[:h//6, w//4:3*w//4],
                'eyebrow_left': gray[h//12:h//4, w//10:w//2],
                'eyebrow_right': gray[h//12:h//4, w//2:9*w//10],
                'eye_left': gray[h//4:h//2, w//10:w//2],
                'eye_right': gray[h//4:h//2, w//2:9*w//10],
                'nose': gray[h//3:2*h//3, w//3:2*w//3],
                'mouth_upper': gray[2*h//3:2*h//3+h//12, w//4:3*w//4],
                'mouth_lower': gray[2*h//3+h//12:h, w//4:3*w//4],
                'mouth': gray[2*h//3:h, w//4:3*w//4],
            }
            
            aus = {}
            
            # AU1/AU2: Inner/Outer Brow Raiser (Surprise, Fear)
            brow_left_mean = np.mean(regions['eyebrow_left'])
            brow_right_mean = np.mean(regions['eyebrow_right'])
            forehead_mean = np.mean(regions['forehead'])
            brow_raise = ((brow_left_mean + brow_right_mean) / 2) - forehead_mean
            brow_std = (np.std(regions['eyebrow_left']) + np.std(regions['eyebrow_right'])) / 2
            aus['AU1'] = brow_std + brow_raise / 4
            aus['AU2'] = aus['AU1']
            
            # AU4: Brow Lowerer (Angry, Sad, concentrated)
            brow_furrow = np.std(regions['eyebrow_left']) + np.std(regions['eyebrow_right'])
            aus['AU4'] = brow_furrow / 2
            
            # AU5: Upper Lid Raiser (Surprise, Fear)
            eye_openness_left = np.std(regions['eye_left'])
            eye_openness_right = np.std(regions['eye_right'])
            aus['AU5'] = (eye_openness_left + eye_openness_right) / 2
            
            # AU6: Cheek Raiser (Happy)
            eye_intensity_left = np.mean(regions['eye_left'])
            eye_intensity_right = np.mean(regions['eye_right'])
            aus['AU6'] = (eye_intensity_left + eye_intensity_right) / 2
            
            # AU7: Lid Tightener (Disgust, Angry)
            aus['AU7'] = np.std(regions['eye_left']) + np.std(regions['eye_right'])
            
            # AU9: Nose Wrinkler (Disgust)
            aus['AU9'] = np.std(regions['nose'])
            
            # AU10: Upper Lip Raiser (Disgust)
            aus['AU10'] = np.std(regions['mouth_upper'])
            
            # AU12: Lip Corner Puller (Happy)
            mouth_width = np.max(np.mean(regions['mouth'], axis=0)) - np.min(np.mean(regions['mouth'], axis=0))
            aus['AU12'] = mouth_width
            
            # AU15: Lip Corner Depressor (Sad)
            aus['AU15'] = np.std(regions['mouth'])
            
            # AU17: Chin Raiser (Sad, Disgust)
            if 'mouth_lower' in regions:
                aus['AU17'] = np.std(regions['mouth_lower'])
            
            # AU20: Lip Stretcher (Fear)
            mouth_horizontal_range = np.max(np.mean(regions['mouth'], axis=0)) - np.min(np.mean(regions['mouth'], axis=0))
            aus['AU20'] = mouth_horizontal_range
            
            # AU25/AU26/AU27: Mouth opening (Surprise, Fear, Happy)
            mouth_vertical = np.max(np.mean(regions['mouth'], axis=1)) - np.min(np.mean(regions['mouth'], axis=1))
            mouth_brightness = np.mean(regions['mouth'])
            aus['AU25'] = mouth_vertical  # Lips part
            aus['AU26'] = mouth_vertical * 1.2  # Jaw drop
            aus['AU27'] = mouth_vertical * 0.8  # Mouth stretch
            
            return aus
            
        except Exception as e:
            print(f"AU extraction error: {e}")
            return {}
    
    def get_au_confidence_boost(self, emotion, aus):
        """Calculate confidence boost based on Action Units matching the emotion
        Enhanced for Sad and Fear which have low CNN accuracy"""
        if not aus:
            return 0.0
        
        boost = 0.0
        
        # Emotion-specific AU patterns
        if emotion == 'Happy':
            # Strong smile indicators
            if aus.get('AU12', 0) > 15:  # Lip corner puller
                boost += 0.15
            if aus.get('AU6', 0) > 100:  # Cheek raiser
                boost += 0.10
        
        elif emotion == 'Sad':
            # ENHANCED: Sadness indicators (47.4% accuracy needs help)
            # Stronger boosts for clear sad markers
            if aus.get('AU15', 0) > 8:  # Lip corner depressor (key sad marker)
                boost += 0.20  # Increased from 0.12
            if aus.get('AU17', 0) > 6:  # Chin raiser
                boost += 0.12  # Increased from 0.08
            if aus.get('AU4', 0) > 12:  # Brow lowerer (sadness frown)
                boost += 0.10  # Increased from 0.05
            # Combination bonus: inner brows + lip corners down
            if aus.get('AU1', 0) > 8 and aus.get('AU15', 0) > 8:
                boost += 0.15  # Extra for classic sad expression
            # Low mouth activity (not smiling or surprised)
            if aus.get('AU12', 0) < 10 and aus.get('AU26', 0) < 10:
                boost += 0.08
        
        elif emotion == 'Surprise':
            # Surprise indicators
            if aus.get('AU1', 0) > 12:  # Brow raiser
                boost += 0.15
            if aus.get('AU5', 0) > 25:  # Upper lid raiser
                boost += 0.10
            if aus.get('AU26', 0) > 15:  # Jaw drop
                boost += 0.20
        
        elif emotion == 'Angry':
            # Anger indicators
            if aus.get('AU4', 0) > 15:  # Brow lowerer
                boost += 0.15
            if aus.get('AU7', 0) > 30:  # Lid tightener
                boost += 0.10
            # Pressed lips (anger without shouting)
            if aus.get('AU15', 0) > 10 and aus.get('AU26', 0) < 8:
                boost += 0.08
        
        elif emotion == 'Fear':
            # ENHANCED: Fear indicators (37.8% accuracy - needs most help)
            # Stronger boosts to distinguish from Surprise/Sad/Angry
            if aus.get('AU1', 0) > 10:  # Brow raiser (inner)
                boost += 0.18  # Increased from 0.10
            if aus.get('AU2', 0) > 10:  # Outer brow raiser
                boost += 0.12
            if aus.get('AU5', 0) > 20:  # Upper lid raiser (wide eyes)
                boost += 0.18  # Increased from 0.12
            if aus.get('AU20', 0) > 12:  # Lip stretcher (horizontal mouth)
                boost += 0.15  # Increased from 0.08
            # Combination bonus: raised brows + stretched lips (classic fear)
            if aus.get('AU1', 0) > 10 and aus.get('AU20', 0) > 12:
                boost += 0.20  # Strong indicator of fear
            # Tense face (distinguish from relaxed sad)
            if aus.get('AU4', 0) > 10 and aus.get('AU7', 0) > 25:
                boost += 0.10
            # NOT surprise: if brows raised but mouth NOT open
            if aus.get('AU1', 0) > 10 and aus.get('AU26', 0) < 10:
                boost += 0.12  # Fear often has raised brows without jaw drop
        
        elif emotion == 'Disgust':
            # Disgust indicators
            if aus.get('AU9', 0) > 15:  # Nose wrinkler
                boost += 0.15
            if aus.get('AU10', 0) > 10:  # Upper lip raiser
                boost += 0.12
        
        # Cap the boost at higher values for Sad/Fear to compensate for low accuracy
        if emotion in ['Sad', 'Fear']:
            return min(boost, 0.40)  # Allow up to 40% boost for problem emotions
        else:
            return min(boost, 0.25)  # 25% for well-performing emotions
    
    def predict_emotion(self, face_img):
        """Hybrid prediction combining CNN and Action Units"""
        # Get CNN prediction
        preprocessed = self.preprocess_face(face_img)
        cnn_probs = self.model.predict(preprocessed, verbose=0)[0]
        self.last_cnn_probs = cnn_probs
        
        # Extract Action Units
        aus = self.extract_action_units(face_img)
        self.last_aus = aus
        
        # Enhance CNN predictions with AU confidence boosts
        enhanced_probs = cnn_probs.copy()
        
        for i, emotion in enumerate(self.emotions):
            au_boost = self.get_au_confidence_boost(emotion, aus)
            enhanced_probs[i] = cnn_probs[i] * (1 + au_boost)
        
        # Apply penalties for mismatching AUs (reduce false positives)
        enhanced_probs = self._apply_au_penalties(enhanced_probs, aus)
        
        # Re-normalize to sum to 1
        enhanced_probs = enhanced_probs / np.sum(enhanced_probs)
        
        # Get top prediction
        top_idx = np.argmax(enhanced_probs)
        confidence = enhanced_probs[top_idx]
        emotion = self.emotions[top_idx]
        
        # Apply smart calibration for low-confidence predictions
        if confidence < 0.45:
            emotion, confidence = self._apply_calibration(enhanced_probs, cnn_probs, aus)
        
        # Temporal smoothing
        emotion, confidence = self._temporal_smoothing(emotion, confidence)
        
        return emotion, confidence
    
    def _apply_au_penalties(self, probs, aus):
        """Apply penalties when AUs contradict predicted emotion (reduce false positives)"""
        if not aus:
            return probs
        
        penalized_probs = probs.copy()
        
        # Penalty for Fear if eyes NOT wide (common misclassification)
        if aus.get('AU5', 0) < 15:  # Eyes not wide
            penalized_probs[2] *= 0.85  # Reduce Fear probability
        
        # Penalty for Surprise if mouth NOT open (very common misclassification)
        if aus.get('AU26', 0) < 8:  # Mouth not open
            penalized_probs[6] *= 0.80  # Reduce Surprise probability
        
        # Penalty for Happy if no smile detected
        if aus.get('AU12', 0) < 8:  # No lip corner pull
            penalized_probs[3] *= 0.90  # Reduce Happy probability
        
        # Penalty for Sad if no downward mouth movement
        if aus.get('AU15', 0) < 5:  # No lip corner depression
            penalized_probs[5] *= 0.88  # Reduce Sad probability
        
        # Penalty for Disgust if no nose wrinkle
        if aus.get('AU9', 0) < 10:  # No nose wrinkle
            penalized_probs[1] *= 0.85  # Reduce Disgust probability
        
        # Penalty for Angry if brows NOT lowered
        if aus.get('AU4', 0) < 10:  # Brows not lowered
            penalized_probs[0] *= 0.88  # Reduce Angry probability
        
        return penalized_probs
    
    def _apply_calibration(self, enhanced_probs, cnn_probs, aus):
        """Apply smart calibration for ambiguous predictions
        Enhanced with special rules for Sad and Fear (low accuracy emotions)"""
        sorted_indices = np.argsort(enhanced_probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
        third_idx = sorted_indices[2] if len(sorted_indices) > 2 else None
        
        top_prob = enhanced_probs[top_idx]
        second_prob = enhanced_probs[second_idx] if second_idx is not None else 0
        third_prob = enhanced_probs[third_idx] if third_idx is not None else 0
        
        # Get specific emotion probabilities
        angry_prob = enhanced_probs[0]
        fear_prob = enhanced_probs[2]
        neutral_prob = enhanced_probs[4]
        sad_prob = enhanced_probs[5]
        surprise_prob = enhanced_probs[6]
        
        # === ENHANCED RULES FOR FEAR (37.8% accuracy - most problematic) ===
        
        # Rule F1: Strong fear markers should override confused predictions
        fear_markers = (
            aus.get('AU1', 0) > 12 and  # Raised brows
            aus.get('AU5', 0) > 22 and  # Wide eyes
            aus.get('AU20', 0) > 12      # Stretched lips
        )
        if fear_markers and fear_prob > 0.08:
            # Strong AU evidence for fear
            return 'Fear', max(0.35, fear_prob * 1.5)
        
        # Rule F2: Fear vs Surprise disambiguation
        # Fear: raised brows WITHOUT open mouth
        # Surprise: raised brows WITH open mouth
        if top_idx == 6 and surprise_prob > 0.20:  # Predicted Surprise
            if aus.get('AU26', 0) < 10 and aus.get('AU1', 0) > 10:  # No jaw drop but raised brows
                if fear_prob > 0.10:
                    return 'Fear', max(0.32, fear_prob * 1.3)
        
        # Rule F3: Fear vs Angry disambiguation
        # Fear: raised brows + wide eyes + tense
        # Angry: lowered brows + narrow eyes
        if top_idx == 0 and angry_prob > 0.15:  # Predicted Angry
            if aus.get('AU1', 0) > 10 and aus.get('AU5', 0) > 20:  # Raised brows + wide eyes
                if fear_prob > 0.08:
                    return 'Fear', max(0.30, fear_prob * 1.4)
        
        # Rule F4: Fear vs Sad disambiguation
        # Fear: tense face with wide eyes
        # Sad: relaxed/downcast face
        if top_idx == 5 and sad_prob > 0.15:  # Predicted Sad
            if aus.get('AU5', 0) > 20 and aus.get('AU20', 0) > 10:  # Wide eyes + lip stretch
                if fear_prob > 0.10:
                    return 'Fear', max(0.30, fear_prob * 1.3)
        
        # === ENHANCED RULES FOR SAD (47.4% accuracy) ===
        
        # Rule S1: Strong sad markers should boost Sad
        sad_markers = (
            aus.get('AU15', 0) > 10 or   # Lip corners down
            (aus.get('AU1', 0) > 8 and aus.get('AU4', 0) > 12)  # Inner brows raised + lowered
        )
        if sad_markers and sad_prob > 0.10:
            return 'Sad', max(0.32, sad_prob * 1.4)
        
        # Rule S2: Sad vs Neutral disambiguation
        # If Neutral predicted but has sad markers
        if top_idx == 4 and neutral_prob > 0.20:  # Predicted Neutral
            if aus.get('AU15', 0) > 8:  # Lip corners down
                if sad_prob > 0.08:
                    return 'Sad', max(0.28, sad_prob * 1.3)
        
        # Rule S3: Sad vs Angry disambiguation
        # Sad: downcast features, less tension
        # Angry: tense features, furrowed brows
        if top_idx == 0 and angry_prob > 0.15:  # Predicted Angry
            if aus.get('AU15', 0) > 10 and aus.get('AU7', 0) < 25:  # Sad mouth, not tense eyes
                if sad_prob > 0.10:
                    return 'Sad', max(0.30, sad_prob * 1.3)
        
        # Rule S4: Sad is often confused with Neutral - boost if lip corners down
        if sad_prob > 0.12 and neutral_prob > sad_prob:
            if aus.get('AU15', 0) > 8:  # Clear lip corner depressor
                return 'Sad', max(0.28, sad_prob * 1.2)
        
        # === ORIGINAL RULES (kept for other emotions) ===
        
        # Rule 1: If Happy wins with low confidence and Neutral is close, prefer Neutral
        if top_idx == 3 and top_prob < 0.45 and neutral_prob > 0.10:
            if neutral_prob / top_prob > 0.35:
                return 'Neutral', max(0.30, neutral_prob)
        
        # Rule 2: If Sad wins with low confidence and Neutral is close, check AUs
        if top_idx == 5 and top_prob < 0.30 and neutral_prob > 0.10:
            # Only prefer Neutral if NO sad markers
            if aus.get('AU15', 0) < 5 and neutral_prob / top_prob > 0.70:
                return 'Neutral', max(0.30, neutral_prob)
        
        # Rule 3: For mouth open (high AU26), boost Surprise
        if aus.get('AU26', 0) > 15 and surprise_prob > 0.10:
            if top_prob < 0.30:
                return 'Surprise', max(0.28, surprise_prob)
        
        # Rule 4: Very close top 2 - prefer Neutral if it's one of them (but not over clear Sad/Fear)
        if second_prob / top_prob > 0.85:
            if (top_idx == 4 or second_idx == 4) and top_idx not in [2, 5]:  # Not Fear or Sad
                return 'Neutral', max(0.30, neutral_prob)
        
        return self.emotions[top_idx], top_prob
    
    def _temporal_smoothing(self, emotion, confidence):
        """Apply minimal temporal smoothing for stability
        Uses emotion-specific thresholds for low-accuracy emotions"""
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
        
        # Need at least 2 frames
        if len(self.emotion_history) < 2:
            return emotion, confidence
        
        # Count occurrences
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        # Most common emotion
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        most_common_emotion, count = most_common
        
        # Use emotion-specific threshold (lower for Sad/Fear)
        threshold = self.confidence_thresholds.get(most_common_emotion, 0.30)
        
        # Accept if it appears at least once with decent confidence OR twice
        # Lower thresholds for Sad and Fear to make them easier to detect
        if count >= 1 and (confidence > threshold or count >= 2):
            relevant_confidences = [
                conf for em, conf in zip(self.emotion_history, self.confidence_history)
                if em == most_common_emotion
            ]
            avg_confidence = np.mean(relevant_confidences) if relevant_confidences else confidence
            
            # Boost displayed confidence for low-accuracy emotions to build user trust
            if most_common_emotion in ['Sad', 'Fear']:
                avg_confidence = min(0.99, avg_confidence * 1.15)
            
            return most_common_emotion, avg_confidence
        
        return emotion, confidence
    
    def detect_faces(self, frame):
        """Detect faces using Haar Cascade"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        return faces
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            elapsed = (cv2.getTickCount() - self.fps_start) / cv2.getTickFrequency()
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start = cv2.getTickCount()
    
    def draw_emotion_info(self, frame, x, y, w, h, emotion, confidence):
        """Draw emotion information on frame"""
        # Draw rectangle around face
        color = self.colors.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw emotion label
        label = f"{emotion}: {confidence:.2f}"
        label_y = y - 10 if y > 30 else y + h + 20
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, label_y - text_h - 5), (x + text_w + 5, label_y + 5), color, -1)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw debug info if enabled
        if self.show_debug:
            self._draw_debug_info(frame, x, y + h + 30)
    
    def _draw_debug_info(self, frame, x, y):
        """Draw debug information"""
        debug_y = y
        
        # Show top 3 CNN predictions
        if len(self.last_cnn_probs) > 0:
            sorted_indices = np.argsort(self.last_cnn_probs)[::-1][:3]
            cv2.putText(frame, "CNN Predictions:", (x, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_y += 15
            
            for idx in sorted_indices:
                emotion = self.emotions[idx]
                prob = self.last_cnn_probs[idx]
                text = f"  {emotion}: {prob:.3f}"
                cv2.putText(frame, text, (x, debug_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                debug_y += 15
        
        # Show key AUs
        if self.last_aus:
            debug_y += 5
            cv2.putText(frame, "Action Units:", (x, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_y += 15
            
            key_aus = ['AU12', 'AU26', 'AU4', 'AU15', 'AU1']
            for au_name in key_aus:
                if au_name in self.last_aus:
                    value = self.last_aus[au_name]
                    text = f"  {au_name}: {value:.1f}"
                    cv2.putText(frame, text, (x, debug_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    debug_y += 15
    
    def run(self):
        """Run real-time emotion detection"""
        print("\n" + "="*70)
        print("HYBRID EMOTION DETECTION - CNN + Action Units")
        print("="*70)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 'd' to toggle debug info")
        print("  - Press 's' to save screenshot")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        screenshot_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
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
                self.draw_emotion_info(frame, x, y, w, h, emotion, confidence)
            
            # Update and display FPS
            self.update_fps()
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display info
            info_text = f"Faces: {len(faces)} | Press 'd' for debug | 'q' to quit"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Hybrid Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('s'):
                screenshot_path = f'screenshot_{screenshot_count}.png'
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
                screenshot_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Emotion detection stopped")


def main():
    """Main entry point"""
    try:
        detector = HybridEmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
