#!/usr/bin/env python3
"""
Enhanced Hybrid Emotion Detection - Multi-Modal CNN with Facial Landmarks
Uses the enhanced CNN model with:
- Image CNN branch
- Geometric features (40 dimensions)
- Action Units features (20 dimensions)
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    print("⚠️ dlib not available - will use OpenCV for face detection only")
    DLIB_AVAILABLE = False
import tensorflow as tf
# Keras is now separate from TensorFlow in TF 2.16+
try:
    from tensorflow import keras
except ImportError:
    import keras

# Configure GPU/MPS (Metal Performance Shaders for Apple Silicon)
print("=" * 60)
print("GPU Configuration:")
print("=" * 60)

# Check available devices
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("ℹ️ No GPU found, checking for Apple Silicon MPS...")
    # For Apple Silicon Macs, check MPS (Metal Performance Shaders)
    try:
        # Set memory growth for MPS if available
        if hasattr(tf.config, 'list_physical_devices'):
            mps_devices = tf.config.list_physical_devices('MPS')
            if mps_devices:
                print(f"✅ Found Apple Metal GPU (MPS): {len(mps_devices)} device(s)")
                print("   Using Metal Performance Shaders for acceleration")
            else:
                print("ℹ️ No MPS device found, will use CPU")
    except Exception as e:
        print(f"ℹ️ MPS check failed: {e}, using CPU")

# Print TensorFlow build info
if tf.test.is_built_with_cuda():
    print("✅ TensorFlow built with CUDA support")
elif hasattr(tf.test, 'is_built_with_rocm') and tf.test.is_built_with_rocm():
    print("✅ TensorFlow built with ROCm support")
else:
    print("ℹ️ TensorFlow built without CUDA (may support MPS or CPU only)")

print("=" * 60)


class FacialLandmarkExtractor:
    """Extract facial landmarks and compute geometric and AU features"""
    
    def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat'):
        """Initialize dlib face detector and landmark predictor"""
        if not DLIB_AVAILABLE:
            self.detector = None
            self.predictor = None
            self.landmarks_available = False
            print("⚠️ dlib not available - facial landmarks will use fallback mode")
            return
            
        self.detector = dlib.get_frontal_face_detector()
        
        try:
            self.predictor = dlib.shape_predictor(predictor_path)
            self.landmarks_available = True
            print(f"✅ Loaded facial landmark predictor: {predictor_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load landmark predictor: {e}")
            print("   Facial landmarks will use fallback mode")
            self.landmarks_available = False
    
    def extract_landmarks(self, image):
        """Extract 68 facial landmarks from image"""
        if not self.landmarks_available:
            return None
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect face
            faces = self.detector(gray)
            
            if len(faces) == 0:
                return None
            
            # Use first detected face
            face = faces[0]
            
            # Predict landmarks
            landmarks = self.predictor(gray, face)
            
            # Convert to numpy array
            coords = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            return coords
            
        except Exception as e:
            return None
    
    def compute_geometric_features(self, landmarks):
        """Compute 40 geometric features from landmarks
        Returns: numpy array of shape (40,)
        """
        if landmarks is None or len(landmarks) != 68:
            return np.zeros(40)
        
        try:
            features = []
            
            # 1. Eye Aspect Ratio (EAR) - Left eye
            left_eye = landmarks[36:42]
            ear_left = self._compute_ear(left_eye)
            features.append(ear_left)
            
            # 2. Eye Aspect Ratio (EAR) - Right eye
            right_eye = landmarks[42:48]
            ear_right = self._compute_ear(right_eye)
            features.append(ear_right)
            
            # 3. Mouth Aspect Ratio (MAR)
            mouth = landmarks[48:68]
            mar = self._compute_mar(mouth)
            features.append(mar)
            
            # 4-5. Eyebrow heights (relative to eyes)
            left_eyebrow = landmarks[17:22]
            right_eyebrow = landmarks[22:27]
            left_brow_height = np.mean(left_eyebrow[:, 1]) - np.mean(left_eye[:, 1])
            right_brow_height = np.mean(right_eyebrow[:, 1]) - np.mean(right_eye[:, 1])
            features.extend([left_brow_height, right_brow_height])
            
            # 6-10. Inter-landmark distances (normalized)
            face_width = np.linalg.norm(landmarks[0] - landmarks[16])
            if face_width == 0:
                face_width = 1.0
            
            # Eye centers
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            
            # Distance between eyes
            eye_distance = np.linalg.norm(left_eye_center - right_eye_center) / face_width
            features.append(eye_distance)
            
            # Mouth center
            mouth_center = np.mean(mouth, axis=0)
            
            # Distance from eyes to mouth
            left_eye_to_mouth = np.linalg.norm(left_eye_center - mouth_center) / face_width
            right_eye_to_mouth = np.linalg.norm(right_eye_center - mouth_center) / face_width
            features.extend([left_eye_to_mouth, right_eye_to_mouth])
            
            # Nose tip to mouth
            nose_tip = landmarks[33]
            nose_to_mouth = np.linalg.norm(nose_tip - mouth_center) / face_width
            features.append(nose_to_mouth)
            
            # Face height (nose bridge to chin)
            face_height = np.linalg.norm(landmarks[27] - landmarks[8]) / face_width
            features.append(face_height)
            
            # 11-20. Angular features (10 angles)
            # Eyebrow angles
            left_brow_angle = self._compute_angle(left_eyebrow[0], left_eyebrow[2], left_eyebrow[4])
            right_brow_angle = self._compute_angle(right_eyebrow[0], right_eyebrow[2], right_eyebrow[4])
            features.extend([left_brow_angle, right_brow_angle])
            
            # Eye angles
            left_eye_angle = self._compute_angle(left_eye[0], left_eye[3], left_eye[5])
            right_eye_angle = self._compute_angle(right_eye[0], right_eye[3], right_eye[5])
            features.extend([left_eye_angle, right_eye_angle])
            
            # Mouth corners angle
            mouth_angle = self._compute_angle(mouth[0], mouth[6], mouth[12])
            features.append(mouth_angle)
            
            # Jaw angles (left and right)
            left_jaw_angle = self._compute_angle(landmarks[0], landmarks[4], landmarks[8])
            right_jaw_angle = self._compute_angle(landmarks[16], landmarks[12], landmarks[8])
            features.extend([left_jaw_angle, right_jaw_angle])
            
            # Nose angles
            nose_bridge_angle = self._compute_angle(landmarks[27], landmarks[30], landmarks[33])
            nose_tip_angle = self._compute_angle(landmarks[31], landmarks[33], landmarks[35])
            features.extend([nose_bridge_angle, nose_tip_angle])
            
            # 21-30. Facial ratios (10 ratios)
            # Eye width ratios
            left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3]) / face_width
            right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3]) / face_width
            features.extend([left_eye_width, right_eye_width])
            
            # Mouth width
            mouth_width = np.linalg.norm(mouth[0] - mouth[6]) / face_width
            features.append(mouth_width)
            
            # Mouth height
            mouth_top = np.mean(mouth[13:16], axis=0)
            mouth_bottom = np.mean(mouth[16:20], axis=0)
            mouth_height = np.linalg.norm(mouth_top - mouth_bottom) / face_width
            features.append(mouth_height)
            
            # Nose width
            nose_width = np.linalg.norm(landmarks[31] - landmarks[35]) / face_width
            features.append(nose_width)
            
            # Face aspect ratio
            face_aspect = face_height / face_width if face_width > 0 else 0
            features.append(face_aspect)
            
            # Eyebrow thickness (distance between inner points)
            left_brow_thickness = np.linalg.norm(left_eyebrow[0] - left_eyebrow[1]) / face_width
            right_brow_thickness = np.linalg.norm(right_eyebrow[3] - right_eyebrow[4]) / face_width
            features.extend([left_brow_thickness, right_brow_thickness])
            
            # Chin prominence
            chin_to_nose = np.linalg.norm(landmarks[8] - landmarks[33]) / face_width
            features.append(chin_to_nose)
            
            # Cheek width (distance from face center to jaw)
            face_center = np.mean(landmarks, axis=0)
            left_cheek = np.linalg.norm(face_center - landmarks[4]) / face_width
            right_cheek = np.linalg.norm(face_center - landmarks[12]) / face_width
            features.extend([left_cheek, right_cheek])
            
            # 31-40. Additional normalized distances (10 more)
            # Upper face distances
            left_brow_to_eye = np.linalg.norm(left_eyebrow[2] - left_eye_center) / face_width
            right_brow_to_eye = np.linalg.norm(right_eyebrow[2] - right_eye_center) / face_width
            features.extend([left_brow_to_eye, right_brow_to_eye])
            
            # Nose bridge to eyes
            nose_bridge = landmarks[27]
            nose_to_left_eye = np.linalg.norm(nose_bridge - left_eye_center) / face_width
            nose_to_right_eye = np.linalg.norm(nose_bridge - right_eye_center) / face_width
            features.extend([nose_to_left_eye, nose_to_right_eye])
            
            # Mouth to chin
            mouth_to_chin = np.linalg.norm(mouth_center - landmarks[8]) / face_width
            features.append(mouth_to_chin)
            
            # Jaw width (distance between jaw points)
            jaw_width_top = np.linalg.norm(landmarks[2] - landmarks[14]) / face_width
            jaw_width_mid = np.linalg.norm(landmarks[4] - landmarks[12]) / face_width
            jaw_width_bottom = np.linalg.norm(landmarks[6] - landmarks[10]) / face_width
            features.extend([jaw_width_top, jaw_width_mid, jaw_width_bottom])
            
            # Face symmetry (left vs right distances)
            left_symmetry = np.linalg.norm(landmarks[0] - face_center) / face_width
            right_symmetry = np.linalg.norm(landmarks[16] - face_center) / face_width
            features.extend([left_symmetry, right_symmetry])
            
            features_array = np.array(features, dtype=np.float32)
            
            # Ensure exactly 40 features
            if len(features_array) != 40:
                print(f"Warning: Expected 40 features, got {len(features_array)}")
                features_array = np.zeros(40, dtype=np.float32)
            
            return features_array
            
        except Exception as e:
            return np.zeros(40)
    
    def extract_action_unit_features(self, landmarks):
        """Extract 20 Action Unit features based on FACS
        Returns: numpy array of shape (20,)
        """
        if landmarks is None or len(landmarks) != 68:
            return np.zeros(20)
        
        try:
            au_features = []
            
            # Get landmark groups
            left_eyebrow = landmarks[17:22]
            right_eyebrow = landmarks[22:27]
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            nose = landmarks[27:36]
            mouth = landmarks[48:68]
            jaw = landmarks[0:17]
            
            # Compute face normalization factors
            face_width = np.linalg.norm(landmarks[0] - landmarks[16])
            if face_width == 0:
                face_width = 1.0
            
            # AU1: Inner Brow Raiser
            inner_brow_left = landmarks[19]
            inner_brow_right = landmarks[24]
            brow_center = (inner_brow_left + inner_brow_right) / 2
            eye_center_y = (np.mean(left_eye[:, 1]) + np.mean(right_eye[:, 1])) / 2
            au1 = (eye_center_y - brow_center[1]) / face_width
            au_features.append(au1)
            
            # AU2: Outer Brow Raiser
            outer_brow_left = landmarks[17]
            outer_brow_right = landmarks[26]
            au2 = (outer_brow_left[1] + outer_brow_right[1]) / 2 / face_width
            au_features.append(au2)
            
            # AU4: Brow Lowerer
            brow_y = np.mean([np.mean(left_eyebrow[:, 1]), np.mean(right_eyebrow[:, 1])])
            au4 = (brow_y - eye_center_y) / face_width
            au_features.append(au4)
            
            # AU5: Upper Lid Raiser (eye opening)
            left_eye_height = np.linalg.norm(left_eye[1] - left_eye[5]) / face_width
            right_eye_height = np.linalg.norm(right_eye[1] - right_eye[5]) / face_width
            au5 = (left_eye_height + right_eye_height) / 2
            au_features.append(au5)
            
            # AU6: Cheek Raiser
            left_cheek_y = landmarks[29][1]  # Nose side
            left_eye_bottom_y = np.mean(left_eye[4:6], axis=0)[1]
            au6 = (left_eye_bottom_y - left_cheek_y) / face_width
            au_features.append(au6)
            
            # AU7: Lid Tightener (eye squint)
            left_eye_width = np.linalg.norm(left_eye[0] - left_eye[3]) / face_width
            right_eye_width = np.linalg.norm(right_eye[0] - right_eye[3]) / face_width
            au7 = (left_eye_width + right_eye_width) / 2
            au_features.append(au7)
            
            # AU9: Nose Wrinkler
            nose_bridge = landmarks[27]
            nose_tip = landmarks[33]
            au9 = np.linalg.norm(nose_bridge - nose_tip) / face_width
            au_features.append(au9)
            
            # AU10: Upper Lip Raiser
            upper_lip_center = landmarks[51]
            nose_bottom = landmarks[33]
            au10 = np.linalg.norm(upper_lip_center - nose_bottom) / face_width
            au_features.append(au10)
            
            # AU12: Lip Corner Puller (smile)
            mouth_left = landmarks[48]
            mouth_right = landmarks[54]
            mouth_width = np.linalg.norm(mouth_left - mouth_right) / face_width
            au12 = mouth_width
            au_features.append(au12)
            
            # AU15: Lip Corner Depressor (frown)
            mouth_center_y = np.mean(mouth[:, 1])
            lip_corner_left_y = landmarks[48][1]
            lip_corner_right_y = landmarks[54][1]
            au15 = (lip_corner_left_y + lip_corner_right_y) / 2 - mouth_center_y
            au15 = au15 / face_width
            au_features.append(au15)
            
            # AU17: Chin Raiser
            chin = landmarks[8]
            lower_lip = landmarks[57]
            au17 = np.linalg.norm(chin - lower_lip) / face_width
            au_features.append(au17)
            
            # AU20: Lip Stretcher
            mouth_corners_dist = np.linalg.norm(landmarks[48] - landmarks[54])
            mouth_center = np.mean(mouth, axis=0)
            au20 = mouth_corners_dist / face_width
            au_features.append(au20)
            
            # AU23: Lip Tightener
            upper_lip_inner = np.mean(landmarks[61:64], axis=0)
            lower_lip_inner = np.mean(landmarks[65:68], axis=0)
            au23 = np.linalg.norm(upper_lip_inner - lower_lip_inner) / face_width
            au_features.append(au23)
            
            # AU24: Lip Pressor
            upper_lip_thickness = np.linalg.norm(landmarks[51] - landmarks[62]) / face_width
            lower_lip_thickness = np.linalg.norm(landmarks[57] - landmarks[66]) / face_width
            au24 = (upper_lip_thickness + lower_lip_thickness) / 2
            au_features.append(au24)
            
            # AU25: Lips Part
            upper_lip_center = landmarks[62]
            lower_lip_center = landmarks[66]
            au25 = np.linalg.norm(upper_lip_center - lower_lip_center) / face_width
            au_features.append(au25)
            
            # AU26: Jaw Drop
            mouth_top = np.mean(landmarks[61:64], axis=0)
            mouth_bottom = np.mean(landmarks[65:68], axis=0)
            au26 = np.linalg.norm(mouth_top - mouth_bottom) / face_width
            au_features.append(au26)
            
            # AU27: Mouth Stretch
            au27 = mouth_width * au26  # Combine width and opening
            au_features.append(au27)
            
            # Additional AUs (17-20) - derived features
            # AU ratio 1: Eye opening ratio
            eye_ratio = au5 / (au7 + 1e-6)
            au_features.append(eye_ratio)
            
            # AU ratio 2: Brow movement ratio
            brow_ratio = au1 / (abs(au4) + 1e-6)
            au_features.append(brow_ratio)
            
            # AU ratio 3: Mouth aspect ratio
            mouth_ratio = au26 / (au12 + 1e-6)
            au_features.append(mouth_ratio)
            
            au_features_array = np.array(au_features, dtype=np.float32)
            
            # Ensure exactly 20 features
            if len(au_features_array) != 20:
                print(f"Warning: Expected 20 AU features, got {len(au_features_array)}")
                au_features_array = np.zeros(20, dtype=np.float32)
            
            return au_features_array
            
        except Exception as e:
            return np.zeros(20)
    
    def _compute_ear(self, eye_points):
        """Compute Eye Aspect Ratio"""
        vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
        horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def _compute_mar(self, mouth_points):
        """Compute Mouth Aspect Ratio"""
        vertical_1 = np.linalg.norm(mouth_points[13] - mouth_points[19])
        vertical_2 = np.linalg.norm(mouth_points[14] - mouth_points[18])
        vertical_3 = np.linalg.norm(mouth_points[15] - mouth_points[17])
        horizontal = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        if horizontal == 0:
            return 0.0
        
        mar = (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)
        return mar
    
    def _compute_angle(self, p1, p2, p3):
        """Compute angle between three points (in degrees)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / np.pi
        
        return angle


class EnhancedHybridEmotionDetector:
    """Enhanced Hybrid Emotion Detection using multi-modal CNN"""
    
    def __init__(self):
        """Initialize the enhanced detector"""
        # Emotion labels (same order as training)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Colors for each emotion (BGR format)
        self.colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 255, 0),    # Green
            'Fear': (255, 0, 255),     # Magenta
            'Happy': (0, 255, 255),    # Yellow
            'Neutral': (255, 255, 255), # White
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (0, 165, 255)  # Orange
        }
        
        # Initialize landmark extractor
        self.landmark_extractor = FacialLandmarkExtractor()
        
        # Load enhanced CNN model
        self.model = self.load_enhanced_cnn_model()
        
        # Initialize face detection
        self.face_cascade = self.init_face_detection()
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start = cv2.getTickCount()
        self.current_fps = 0
        
        # Temporal smoothing
        self.emotion_history = []
        self.confidence_history = []
        self.history_size = 2
        
        # Confidence thresholds (lower for problematic emotions)
        self.confidence_thresholds = {
            'Happy': 0.30,
            'Surprise': 0.30,
            'Disgust': 0.30,
            'Neutral': 0.30,
            'Angry': 0.28,
            'Sad': 0.25,
            'Fear': 0.22
        }
        
        # Debug info
        self.show_debug = False
        self.last_cnn_probs = []
        self.last_geometric_features = None
        self.last_au_features = None
    
    def load_enhanced_cnn_model(self):
        """Load the enhanced multi-modal CNN model"""
        # Look for enhanced CNN models
        models_dir = Path(__file__).parent.parent.parent / 'models'
        model_files = []
        
        if models_dir.exists():
            # Look for enhanced models (exclude simple and raf_db models)
            for pattern in ['emotion_enhanced_cnn*.h5', 'enhanced_cnn*.h5']:
                model_files.extend(glob.glob(str(models_dir / pattern)))
        
        if not model_files:
            raise FileNotFoundError(
                "❌ No enhanced CNN model found!\n"
                "   Please train the enhanced model first using:\n"
                "   python src/core/train_enhanced_cnn.py"
            )
        
        # Use the newest model
        newest_model = max(model_files, key=os.path.getctime)
        
        try:
            # Load model with compatibility for older Keras versions
            import warnings
            import h5py
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Use TensorFlow's legacy h5 format loading
            with h5py.File(newest_model, 'r') as f:
                # Check if it's a Keras 3.x model
                if 'keras_version' in f.attrs:
                    print(f"ℹ️ Model saved with Keras {f.attrs['keras_version']}")
            
            # Try different loading strategies
            try:
                # Strategy 1: Load with compile=False (works for most TF 2.x models)
                model = keras.models.load_model(newest_model, compile=False)
            except Exception as e1:
                print(f"⚠️ Standard load failed: {str(e1)[:100]}...")
                try:
                    # Strategy 2: Use tf.keras.models specifically
                    import tensorflow as tf
                    model = tf.keras.models.load_model(newest_model, compile=False)
                except Exception as e2:
                    print(f"⚠️ TF Keras load failed: {str(e2)[:100]}...")
                    # Strategy 3: Load weights only if architecture is known
                    raise Exception(
                        f"Failed to load model. The model may have been saved with an incompatible Keras version.\n"
                        f"   Original error: {e1}\n"
                        f"   Please retrain the model with: python src/core/train_enhanced_cnn.py"
                    )
            
            # Recompile the model
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Loaded enhanced CNN model: {newest_model}")
            print(f"   Model inputs: {[inp.name for inp in model.inputs]}")
            print(f"   Model output: {model.output.shape}")
            return model
        except Exception as e:
            raise Exception(f"❌ Failed to load model: {e}")
    
    def init_face_detection(self):
        """Initialize Haar Cascade face detection"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("⚠️ Warning: Could not load Haar Cascade classifier")
            return None
        
        print("✅ Initialized Haar Cascade face detection")
        return face_cascade
    
    def preprocess_face(self, face_img):
        """Preprocess face and extract all features for enhanced CNN
        Returns: dict with 3 inputs: image_input, geometric_input, au_input
        """
        # 1. Preprocess image for CNN (96x96 grayscale)
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        resized = cv2.resize(gray, (96, 96))
        normalized = resized.astype('float32') / 255.0
        image_input = normalized.reshape(1, 96, 96, 1)
        
        # 2. Extract landmarks
        landmarks = self.landmark_extractor.extract_landmarks(face_img)
        
        # 3. Compute geometric features (40 dimensions)
        geometric_features = self.landmark_extractor.compute_geometric_features(landmarks)
        geometric_input = geometric_features.reshape(1, 40)
        self.last_geometric_features = geometric_features
        
        # 4. Compute AU features (20 dimensions)
        au_features = self.landmark_extractor.extract_action_unit_features(landmarks)
        au_input = au_features.reshape(1, 20)
        self.last_au_features = au_features
        
        # Return dict with all three inputs
        return {
            'image_input': image_input,
            'geometric_input': geometric_input,
            'au_input': au_input
        }
    
    def predict_emotion(self, face_img):
        """Predict emotion using enhanced multi-modal CNN"""
        # Preprocess and extract all features
        model_inputs = self.preprocess_face(face_img)
        
        # Get prediction from enhanced model
        probs = self.model.predict(model_inputs, verbose=0)[0]
        self.last_cnn_probs = probs
        
        # Get top prediction
        top_idx = np.argmax(probs)
        confidence = probs[top_idx]
        emotion = self.emotions[top_idx]
        
        # Apply smart calibration for low-confidence predictions
        if confidence < 0.45:
            emotion, confidence = self._apply_calibration(probs, model_inputs)
        
        # Temporal smoothing
        emotion, confidence = self._temporal_smoothing(emotion, confidence)
        
        return emotion, confidence
    
    def _apply_calibration(self, probs, model_inputs):
        """Apply smart calibration for ambiguous predictions"""
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = sorted_indices[0]
        second_idx = sorted_indices[1] if len(sorted_indices) > 1 else None
        
        top_prob = probs[top_idx]
        second_prob = probs[second_idx] if second_idx is not None else 0
        
        # Get specific emotion probabilities
        fear_prob = probs[2]
        neutral_prob = probs[4]
        sad_prob = probs[5]
        surprise_prob = probs[6]
        
        # Get AU features for additional checks
        au_features = self.last_au_features if self.last_au_features is not None else np.zeros(20)
        
        # === ENHANCED RULES FOR FEAR (most problematic) ===
        
        # Rule F1: Strong fear markers from geometric/AU features
        # Check if eyes are wide and brows raised (fear indicators)
        if len(au_features) >= 5:
            au5 = au_features[3]  # Upper Lid Raiser
            au1 = au_features[0]  # Inner Brow Raiser
            
            fear_markers = au5 > 0.05 and au1 > 0.05
            if fear_markers and fear_prob > 0.08:
                return 'Fear', max(0.35, fear_prob * 1.5)
        
        # Rule F2: Fear vs Surprise disambiguation
        if top_idx == 6 and surprise_prob > 0.20:
            if len(au_features) >= 16:
                au26 = au_features[15]  # Jaw Drop
                au1 = au_features[0]    # Brow Raiser
                
                # Fear: raised brows WITHOUT jaw drop
                if au26 < 0.03 and au1 > 0.03 and fear_prob > 0.10:
                    return 'Fear', max(0.32, fear_prob * 1.3)
        
        # === ENHANCED RULES FOR SAD ===
        
        # Rule S1: Sad markers from geometric/AU features
        if len(au_features) >= 11:
            au15 = au_features[9]   # Lip Corner Depressor
            au4 = au_features[2]    # Brow Lowerer
            
            sad_markers = au15 > 0.02 or au4 > 0.03
            if sad_markers and sad_prob > 0.10:
                return 'Sad', max(0.32, sad_prob * 1.4)
        
        # Rule S2: Sad vs Neutral disambiguation
        if top_idx == 4 and neutral_prob > 0.20:
            if len(au_features) >= 10 and au_features[9] > 0.02:  # Lip corners down
                if sad_prob > 0.08:
                    return 'Sad', max(0.28, sad_prob * 1.3)
        
        # === ORIGINAL RULES ===
        
        # Rule 1: Very close top 2 - prefer Neutral if it's one of them
        if second_prob / top_prob > 0.85:
            if (top_idx == 4 or second_idx == 4) and top_idx not in [2, 5]:
                return 'Neutral', max(0.30, neutral_prob)
        
        return self.emotions[top_idx], top_prob
    
    def _temporal_smoothing(self, emotion, confidence):
        """Apply temporal smoothing for stability"""
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
        
        if len(self.emotion_history) < 2:
            return emotion, confidence
        
        # Count occurrences
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        # Most common emotion
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        most_common_emotion, count = most_common
        
        # Use emotion-specific threshold
        threshold = self.confidence_thresholds.get(most_common_emotion, 0.30)
        
        if count >= 1 and (confidence > threshold or count >= 2):
            relevant_confidences = [
                conf for em, conf in zip(self.emotion_history, self.confidence_history)
                if em == most_common_emotion
            ]
            avg_confidence = np.mean(relevant_confidences) if relevant_confidences else confidence
            
            # Boost displayed confidence for low-accuracy emotions
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
            cv2.putText(frame, "Enhanced CNN Predictions:", (x, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_y += 15
            
            for idx in sorted_indices:
                emotion = self.emotions[idx]
                prob = self.last_cnn_probs[idx]
                text = f"  {emotion}: {prob:.3f}"
                cv2.putText(frame, text, (x, debug_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                debug_y += 15
        
        # Show key geometric features
        if self.last_geometric_features is not None:
            debug_y += 5
            cv2.putText(frame, "Geometric Features:", (x, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_y += 15
            
            geo = self.last_geometric_features
            if len(geo) >= 3:
                text = f"  EAR_L:{geo[0]:.3f} EAR_R:{geo[1]:.3f} MAR:{geo[2]:.3f}"
                cv2.putText(frame, text, (x, debug_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                debug_y += 15
        
        # Show key AU features
        if self.last_au_features is not None:
            debug_y += 5
            cv2.putText(frame, "Action Unit Features:", (x, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            debug_y += 15
            
            au = self.last_au_features
            if len(au) >= 10:
                text = f"  AU1:{au[0]:.3f} AU5:{au[3]:.3f} AU12:{au[8]:.3f}"
                cv2.putText(frame, text, (x, debug_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                debug_y += 15
    
    def run(self):
        """Run real-time emotion detection"""
        print("\n" + "="*70)
        print("ENHANCED HYBRID EMOTION DETECTION - Multi-Modal CNN")
        print("="*70)
        print("Architecture:")
        print("  - Image CNN Branch (256 features)")
        print("  - Geometric Features Branch (40 → 64 features)")
        print("  - Action Units Branch (20 → 32 features)")
        print("  - Total: 352 concatenated features → Emotion prediction")
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
            cv2.imshow('Enhanced Hybrid Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('s'):
                screenshot_path = f'screenshot_enhanced_{screenshot_count}.png'
                cv2.imwrite(screenshot_path, frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
                screenshot_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Enhanced emotion detection stopped")


def main():
    """Main entry point"""
    try:
        detector = EnhancedHybridEmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
