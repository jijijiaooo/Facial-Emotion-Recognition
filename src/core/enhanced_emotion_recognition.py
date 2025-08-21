#!/usr/bin/env python3
"""
Enhanced Facial Emotion Recognition with Research-Based Improvements
Incorporates state-of-the-art techniques for improved accuracy
"""

import cv2
import numpy as np
import time
from collections import deque
import threading
import queue
import warnings
warnings.filterwarnings("ignore")

# Try to import advanced libraries
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from tensorflow import keras
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

class EnhancedEmotionRecognizer:
    def __init__(self):
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.colors = {
            'Angry': (0, 0, 255), 'Disgust': (0, 255, 0), 'Fear': (255, 0, 255),
            'Happy': (0, 255, 255), 'Neutral': (255, 255, 255), 'Sad': (255, 0, 0),
            'Surprise': (0, 165, 255)
        }
        
        # Enhanced preprocessing parameters
        self.input_size = (48, 48)
        self.use_data_augmentation = True
        self.use_ensemble = True
        self.temporal_smoothing = True
        
        # Temporal smoothing for stable predictions
        self.emotion_history = deque(maxlen=5)  # Store last 5 predictions
        self.confidence_threshold = 0.6
        
        # Multi-scale face detection
        self.face_scales = [(1.0, 1.0), (0.8, 0.8), (1.2, 1.2)]
        
        # Initialize components
        self.init_face_detection()
        self.init_emotion_models()
        self.init_preprocessing()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start = time.time()
        self.current_fps = 0
    
    def init_face_detection(self):
        """Initialize enhanced face detection with multiple methods"""
        self.face_detectors = []
        
        # Method 1: MediaPipe (primary)
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.7
            )
            self.face_detectors.append(("mediapipe", self.detect_faces_mediapipe))
            print("✅ MediaPipe face detection initialized")
        
        # Method 2: OpenCV DNN (secondary)
        try:
            self.dnn_net = cv2.dnn.readNetFromTensorflow(
                'opencv_face_detector_uint8.pb',
                'opencv_face_detector.pbtxt'
            )
            self.face_detectors.append(("dnn", self.detect_faces_dnn))
            print("✅ OpenCV DNN face detection initialized")
        except:
            pass
        
        # Method 3: Haar Cascade (fallback)
        try:
            # Try multiple paths for the Haar cascade file
            cascade_paths = [
                'haarcascade_frontalface_default.xml',
                'config/haarcascade_frontalface_default.xml',
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            ]
            
            self.face_cascade = None
            for cascade_path in cascade_paths:
                try:
                    cascade = cv2.CascadeClassifier(cascade_path)
                    if not cascade.empty():
                        self.face_cascade = cascade
                        print(f"✅ Haar Cascade loaded from: {cascade_path}")
                        break
                except:
                    continue
            
            if self.face_cascade is not None:
                self.face_detectors.append(("haar", self.detect_faces_haar))
                print("✅ Haar Cascade face detection initialized")
            else:
                print("⚠️ Haar Cascade file not found - skipping")
        except Exception as e:
            print(f"⚠️ Haar Cascade initialization failed: {e}")
    
    def init_emotion_models(self):
        """Initialize multiple emotion models for ensemble prediction"""
        self.models = []
        
        # Primary model: Your trained model
        if TF_AVAILABLE:
            try:
                model = keras.models.load_model('model_file_30epochs.h5')
                self.models.append(("primary", model, 1.0))  # Weight = 1.0
                print("✅ Primary emotion model loaded")
            except Exception as e:
                print(f"Failed to load primary model: {e}")
        
        # Secondary models (if available)
        self.load_additional_models()
    
    def load_additional_models(self):
        """Load additional models for ensemble prediction"""
        # Try to load pre-trained models or create lightweight alternatives
        model_files = [
            ('emotion_model.tflite', 0.8),
            ('emotion_model_quantized.tflite', 0.6),
        ]
        
        for model_file, weight in model_files:
            try:
                import tflite_runtime.interpreter as tflite
                interpreter = tflite.Interpreter(model_path=model_file)
                interpreter.allocate_tensors()
                self.models.append(("tflite", interpreter, weight))
                print(f"✅ Additional model loaded: {model_file}")
            except:
                continue
    
    def init_preprocessing(self):
        """Initialize advanced preprocessing techniques"""
        # Histogram equalization for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Gaussian blur for noise reduction
        self.blur_kernel = (3, 3)
        
        # Data augmentation parameters
        self.augmentation_params = {
            'rotation_range': 10,
            'brightness_range': 0.2,
            'contrast_range': 0.2
        }
    
    def detect_faces_mediapipe(self, frame):
        """Enhanced MediaPipe face detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                # Add padding for better emotion recognition
                padding = 0.1
                x = max(0, int((bbox.xmin - padding) * w))
                y = max(0, int((bbox.ymin - padding) * h))
                width = min(w - x, int((bbox.width + 2*padding) * w))
                height = min(h - y, int((bbox.height + 2*padding) * h))
                
                # Quality check
                if width > 30 and height > 30:
                    confidence = detection.score[0]
                    faces.append((x, y, width, height, confidence))
        
        return faces
    
    def detect_faces_dnn(self, frame):
        """DNN-based face detection for better accuracy"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
        
        return faces
    
    def detect_faces_haar(self, frame):
        """Haar cascade detection with multi-scale"""
        if self.face_cascade is None or self.face_cascade.empty():
            print("⚠️ Haar cascade not available")
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Add confidence score (estimated)
            faces_with_conf = [(x, y, w, h, 0.8) for (x, y, w, h) in faces]
            return faces_with_conf
            
        except Exception as e:
            print(f"⚠️ Haar cascade detection error: {e}")
            return []
    
    def enhanced_preprocessing(self, face_img):
        """Apply research-based preprocessing techniques"""
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img.copy()
        
        # 1. Histogram equalization for lighting normalization
        equalized = self.clahe.apply(gray)
        
        # 2. Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(equalized, self.blur_kernel, 0)
        
        # 3. Resize to model input size
        resized = cv2.resize(blurred, self.input_size)
        
        # 4. Normalization
        normalized = resized.astype(np.float32) / 255.0
        
        # 5. Optional: Add slight contrast enhancement
        contrast_enhanced = cv2.convertScaleAbs(normalized * 255, alpha=1.1, beta=5) / 255.0
        
        return contrast_enhanced
    
    def data_augmentation(self, face_img):
        """Generate augmented versions for ensemble prediction"""
        augmented_images = [face_img]  # Original
        
        if not self.use_data_augmentation:
            return augmented_images
        
        # Slight rotation
        center = (face_img.shape[1]//2, face_img.shape[0]//2)
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]))
            augmented_images.append(rotated)
        
        # Brightness adjustment
        for brightness in [0.9, 1.1]:
            bright = cv2.convertScaleAbs(face_img * 255, alpha=brightness, beta=0) / 255.0
            augmented_images.append(bright)
        
        return augmented_images
    
    def ensemble_prediction(self, face_img):
        """Use multiple models and techniques for robust prediction"""
        if not self.models:
            return "Neutral", 0.5
        
        # Preprocess image
        processed_img = self.enhanced_preprocessing(face_img)
        
        # Generate augmented versions
        augmented_images = self.data_augmentation(processed_img)
        
        all_predictions = []
        
        # Get predictions from all models
        for model_name, model, weight in self.models:
            model_predictions = []
            
            for aug_img in augmented_images:
                try:
                    if model_name == "primary":
                        # Keras model prediction
                        input_data = aug_img.reshape(1, 48, 48, 1)
                        pred = model.predict(input_data, verbose=0)[0]
                        model_predictions.append(pred)
                    
                    elif model_name == "tflite":
                        # TensorFlow Lite prediction
                        input_details = model.get_input_details()
                        output_details = model.get_output_details()
                        
                        input_data = aug_img.reshape(1, 48, 48, 1).astype(np.float32)
                        model.set_tensor(input_details[0]['index'], input_data)
                        model.invoke()
                        pred = model.get_tensor(output_details[0]['index'])[0]
                        model_predictions.append(pred)
                
                except Exception as e:
                    print(f"Model {model_name} prediction failed: {e}")
                    continue
            
            if model_predictions:
                # Average predictions from augmented images
                avg_pred = np.mean(model_predictions, axis=0)
                # Weight by model importance
                weighted_pred = avg_pred * weight
                all_predictions.append(weighted_pred)
        
        if not all_predictions:
            return "Neutral", 0.5
        
        # Ensemble: weighted average of all model predictions
        final_prediction = np.mean(all_predictions, axis=0)
        
        # Get emotion and confidence
        emotion_idx = np.argmax(final_prediction)
        confidence = final_prediction[emotion_idx]
        
        return self.emotions[emotion_idx], confidence
    
    def temporal_smoothing_prediction(self, emotion, confidence):
        """Apply temporal smoothing for stable predictions"""
        if not self.temporal_smoothing:
            return emotion, confidence
        
        # Add current prediction to history
        self.emotion_history.append((emotion, confidence))
        
        if len(self.emotion_history) < 3:
            return emotion, confidence
        
        # Count occurrences of each emotion in recent history
        emotion_counts = {}
        total_confidence = 0
        
        for hist_emotion, hist_conf in self.emotion_history:
            if hist_emotion not in emotion_counts:
                emotion_counts[hist_emotion] = []
            emotion_counts[hist_emotion].append(hist_conf)
            total_confidence += hist_conf
        
        # Find most frequent emotion with high confidence
        best_emotion = emotion
        best_score = 0
        
        for emo, confidences in emotion_counts.items():
            # Score = frequency * average confidence
            score = len(confidences) * np.mean(confidences)
            if score > best_score:
                best_score = score
                best_emotion = emo
        
        # Use smoothed prediction if confidence is high enough
        avg_confidence = total_confidence / len(self.emotion_history)
        if avg_confidence > self.confidence_threshold:
            return best_emotion, avg_confidence
        else:
            return emotion, confidence
    
    def detect_faces(self, frame):
        """Use best available face detection method"""
        for detector_name, detector_func in self.face_detectors:
            try:
                faces = detector_func(frame)
                if faces:
                    return faces, detector_name
            except Exception as e:
                print(f"Face detector {detector_name} failed: {e}")
                continue
        
        return [], "none"
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start = time.time()
    
    def run(self):
        """Main enhanced emotion recognition loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("🚀 Starting Enhanced Emotion Recognition")
        print("🧠 Using research-based improvements for higher accuracy")
        print("Press 'q' to quit, 'i' for info, 't' to toggle temporal smoothing")
        
        show_info = True
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.update_fps()
                
                # Process every 2nd frame for performance
                if frame_count % 2 == 0:
                    # Detect faces
                    faces, detector_used = self.detect_faces(frame)
                    
                    # Process each face
                    for face_data in faces:
                        if len(face_data) == 5:  # x, y, w, h, confidence
                            x, y, w, h, face_conf = face_data
                        else:  # x, y, w, h
                            x, y, w, h = face_data
                            face_conf = 0.8
                        
                        # Extract face region
                        face_img = frame[y:y+h, x:x+w]
                        if face_img.size > 0:
                            # Enhanced emotion prediction
                            emotion, confidence = self.ensemble_prediction(face_img)
                            
                            # Apply temporal smoothing
                            emotion, confidence = self.temporal_smoothing_prediction(emotion, confidence)
                            
                            # Draw results
                            color = self.colors.get(emotion, (255, 255, 255))
                            
                            # Face rectangle with thickness based on confidence
                            thickness = int(2 + confidence * 2)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                            
                            # Emotion label with enhanced formatting
                            label = f"{emotion.upper()}"
                            if confidence > 0.7:
                                label += f" ({confidence:.2f})"
                            
                            # Background for text
                            cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
                            cv2.putText(frame, label, (x+5, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            # Confidence bar
                            bar_width = int(w * confidence)
                            cv2.rectangle(frame, (x, y+h+5), (x+bar_width, y+h+15), color, -1)
                
                # Show enhanced info
                if show_info:
                    info_lines = [
                        f"FPS: {self.current_fps} | Models: {len(self.models)}",
                        f"Detector: {detector_used if 'detector_used' in locals() else 'none'}",
                        f"Temporal Smoothing: {'ON' if self.temporal_smoothing else 'OFF'}",
                        f"Ensemble: {'ON' if self.use_ensemble else 'OFF'}"
                    ]
                    
                    for i, line in enumerate(info_lines):
                        cv2.putText(frame, line, (10, 25 + i*20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display frame
                cv2.imshow("Enhanced Emotion Recognition", frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('i'):
                    show_info = not show_info
                elif key == ord('t'):
                    self.temporal_smoothing = not self.temporal_smoothing
                    print(f"Temporal smoothing: {'ON' if self.temporal_smoothing else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Enhanced emotion recognition stopped")
            print(f"📊 Final FPS: {self.current_fps}")

if __name__ == "__main__":
    recognizer = EnhancedEmotionRecognizer()
    recognizer.run()