#!/usr/bin/env python3
"""
Action Unit Enhanced Training for Emotion Recognition
Incorporates Facial Action Coding System (FACS) for better accuracy
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Try to import advanced libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

class ActionUnitExtractor:
    """Extract facial action units for emotion recognition"""
    
    def __init__(self):
        # Emotion-relevant Action Units mapping
        self.emotion_au_mapping = {
            'Happy': [6, 12, 25],      # Cheek raiser, Lip corner puller, Lips part
            'Sad': [1, 4, 15, 17],     # Inner brow raiser, Brow lowerer, Lip corner depressor, Chin raiser
            'Angry': [4, 5, 7, 23],    # Brow lowerer, Upper lid raiser, Lid tightener, Lip tightener
            'Fear': [1, 2, 4, 5, 20, 26], # Inner brow raiser, Outer brow raiser, Brow lowerer, Upper lid raiser, Lip stretcher, Jaw drop
            'Surprise': [1, 2, 5, 26, 27], # Inner brow raiser, Outer brow raiser, Upper lid raiser, Jaw drop, Mouth stretch
            'Disgust': [9, 15, 16],    # Nose wrinkler, Lip corner depressor, Lower lip depressor
            'Neutral': []              # Baseline state
        }
        
        # Initialize facial landmark detector
        if DLIB_AVAILABLE:
            try:
                self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                self.face_detector = dlib.get_frontal_face_detector()
                self.landmarks_available = True
                print("✅ Facial landmark detection initialized")
            except:
                self.landmarks_available = False
                print("⚠️ Landmark model not found, using geometric features")
        else:
            self.landmarks_available = False
            print("⚠️ dlib not available, using basic features")
    
    def extract_landmarks(self, image):
        """Extract 68 facial landmarks"""
        if not self.landmarks_available:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Use the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        return points
    
    def calculate_au_features(self, landmarks):
        """Calculate Action Unit features from landmarks"""
        if landmarks is None:
            return np.zeros(20)  # Return zero features if no landmarks
        
        features = []
        
        # AU1: Inner Brow Raiser (distance between inner eyebrows)
        inner_brow_dist = np.linalg.norm(landmarks[21] - landmarks[22])
        features.append(inner_brow_dist)
        
        # AU2: Outer Brow Raiser (eyebrow height)
        left_brow_height = landmarks[19][1] - landmarks[37][1]
        right_brow_height = landmarks[24][1] - landmarks[44][1]
        features.extend([left_brow_height, right_brow_height])
        
        # AU4: Brow Lowerer (eyebrow to eye distance)
        left_brow_eye = landmarks[19][1] - landmarks[37][1]
        right_brow_eye = landmarks[24][1] - landmarks[44][1]
        features.extend([left_brow_eye, right_brow_eye])
        
        # AU5: Upper Lid Raiser (eye opening)
        left_eye_opening = landmarks[41][1] - landmarks[37][1]
        right_eye_opening = landmarks[46][1] - landmarks[43][1]
        features.extend([left_eye_opening, right_eye_opening])
        
        # AU6: Cheek Raiser (cheek to eye distance)
        left_cheek_raise = landmarks[31][1] - landmarks[39][1]
        right_cheek_raise = landmarks[35][1] - landmarks[42][1]
        features.extend([left_cheek_raise, right_cheek_raise])
        
        # AU12: Lip Corner Puller (mouth width)
        mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
        features.append(mouth_width)
        
        # AU15: Lip Corner Depressor (mouth corner position)
        left_corner_pos = landmarks[48][1] - landmarks[33][1]
        right_corner_pos = landmarks[54][1] - landmarks[33][1]
        features.extend([left_corner_pos, right_corner_pos])
        
        # AU17: Chin Raiser (chin to lip distance)
        chin_lip_dist = landmarks[57][1] - landmarks[8][1]
        features.append(chin_lip_dist)
        
        # AU20: Lip Stretcher (lip thinning)
        upper_lip_height = landmarks[51][1] - landmarks[62][1]
        lower_lip_height = landmarks[66][1] - landmarks[57][1]
        features.extend([upper_lip_height, lower_lip_height])
        
        # AU25: Lips Part (mouth opening)
        mouth_opening = landmarks[62][1] - landmarks[66][1]
        features.append(mouth_opening)
        
        # AU26: Jaw Drop (jaw position)
        jaw_drop = landmarks[8][1] - landmarks[33][1]
        features.append(jaw_drop)
        
        # Normalize features by face size
        face_height = landmarks[8][1] - landmarks[19][1]
        face_width = landmarks[16][0] - landmarks[0][0]
        face_size = max(face_height, face_width)
        
        if face_size > 0:
            features = [f / face_size for f in features]
        
        return np.array(features)
    
    def extract_geometric_features(self, image):
        """Extract geometric features when landmarks aren't available"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Basic geometric features
        features = []
        
        # Image moments for shape analysis
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            features.extend([cx / gray.shape[1], cy / gray.shape[0]])  # Normalized centroid
        else:
            features.extend([0.5, 0.5])
        
        # Texture features using LBP-like analysis
        h, w = gray.shape
        regions = [
            gray[:h//3, :w//3],      # Top-left (left eyebrow)
            gray[:h//3, w//3:2*w//3], # Top-center (forehead)
            gray[:h//3, 2*w//3:],    # Top-right (right eyebrow)
            gray[h//3:2*h//3, :w//3], # Mid-left (left eye)
            gray[h//3:2*h//3, w//3:2*w//3], # Mid-center (nose)
            gray[h//3:2*h//3, 2*w//3:], # Mid-right (right eye)
            gray[2*h//3:, :w//3],    # Bottom-left (left cheek)
            gray[2*h//3:, w//3:2*w//3], # Bottom-center (mouth)
            gray[2*h//3:, 2*w//3:],  # Bottom-right (right cheek)
        ]
        
        for region in regions:
            if region.size > 0:
                features.extend([
                    np.mean(region) / 255.0,  # Average intensity
                    np.std(region) / 255.0,   # Texture variation
                ])
            else:
                features.extend([0.0, 0.0])
        
        return np.array(features)
    
    def extract_features(self, image):
        """Extract comprehensive facial features"""
        if self.landmarks_available:
            landmarks = self.extract_landmarks(image)
            au_features = self.calculate_au_features(landmarks)
        else:
            au_features = np.zeros(20)
        
        # Always extract geometric features as backup
        geometric_features = self.extract_geometric_features(image)
        
        # Combine features
        combined_features = np.concatenate([au_features, geometric_features])
        return combined_features

class AUEnhancedDataset(Dataset):
    """Dataset that includes Action Unit features"""
    
    def __init__(self, images, labels, au_extractor, transform=None):
        self.images = images
        self.labels = labels
        self.au_extractor = au_extractor
        self.transform = transform
        
        # Pre-extract AU features for efficiency
        print("Extracting Action Unit features...")
        self.au_features = []
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(images)}")
            features = self.au_extractor.extract_features(img)
            self.au_features.append(features)
        self.au_features = np.array(self.au_features)
        
        # Normalize AU features
        self.scaler = StandardScaler()
        self.au_features = self.scaler.fit_transform(self.au_features)
        
        print(f"✅ Extracted {self.au_features.shape[1]} AU features per image")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        au_features = self.au_features[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': torch.FloatTensor(image),
            'au_features': torch.FloatTensor(au_features),
            'label': torch.LongTensor([label])
        }

class MultiModalEmotionNet(nn.Module):
    """Neural network that combines CNN features with Action Unit features"""
    
    def __init__(self, num_classes=7, au_feature_dim=38):
        super(MultiModalEmotionNet, self).__init__()
        
        # CNN branch for image features
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # AU feature branch
        self.au_branch = nn.Sequential(
            nn.Linear(au_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(256 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Weights for CNN and AU features
            nn.Softmax(dim=1)
        )
    
    def forward(self, image, au_features):
        # Extract CNN features
        cnn_features = self.cnn_branch(image)
        
        # Extract AU features
        au_features = self.au_branch(au_features)
        
        # Concatenate features
        combined = torch.cat([cnn_features, au_features], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        
        # Weighted combination
        weighted_cnn = cnn_features * attention_weights[:, 0:1]
        weighted_au = au_features * attention_weights[:, 1:2]
        
        # Final prediction
        final_features = torch.cat([weighted_cnn, weighted_au], dim=1)
        output = self.fusion(final_features)
        
        return output

def create_diverse_training_recommendations():
    """Provide recommendations for creating diverse training datasets"""
    
    recommendations = {
        "Dataset Diversity": [
            "Include multiple age groups (children, adults, elderly)",
            "Balance gender representation",
            "Include different ethnicities and cultural backgrounds",
            "Vary lighting conditions (bright, dim, natural, artificial)",
            "Include different poses (frontal, slight angles)",
            "Add various backgrounds and contexts"
        ],
        
        "Expression Intensity": [
            "Include subtle micro-expressions",
            "Add moderate intensity expressions",
            "Include strong/exaggerated expressions",
            "Capture transition states between emotions",
            "Include mixed emotions (happy-surprised, sad-angry)"
        ],
        
        "Action Unit Coverage": [
            "Ensure all relevant AUs are represented",
            "Include AU combinations for complex emotions",
            "Add asymmetric expressions (one-sided smiles)",
            "Include partial occlusions (hand covering mouth)",
            "Add glasses, facial hair, makeup variations"
        ],
        
        "Data Augmentation": [
            "Rotation: ±15 degrees",
            "Brightness: ±20%",
            "Contrast: ±20%",
            "Horizontal flipping (be careful with asymmetric expressions)",
            "Slight scaling: 0.9-1.1x",
            "Gaussian noise addition"
        ],
        
        "Quality Control": [
            "Manual verification of labels",
            "Remove ambiguous expressions",
            "Ensure consistent labeling criteria",
            "Balance class distributions",
            "Remove duplicate or near-duplicate images"
        ]
    }
    
    return recommendations

def main():
    print("🎭 Action Unit Enhanced Emotion Recognition Training")
    print("=" * 60)
    
    # Initialize AU extractor
    au_extractor = ActionUnitExtractor()
    
    # Show recommendations
    recommendations = create_diverse_training_recommendations()
    
    print("\n📋 Training Dataset Recommendations:")
    print("-" * 40)
    
    for category, items in recommendations.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    print(f"\n🔬 Action Unit Analysis:")
    print("-" * 30)
    
    for emotion, aus in au_extractor.emotion_au_mapping.items():
        if aus:
            print(f"{emotion:10}: AUs {', '.join(map(str, aus))}")
        else:
            print(f"{emotion:10}: Baseline state")
    
    print(f"\n✅ System initialized with:")
    print(f"   • Facial landmarks: {'Available' if au_extractor.landmarks_available else 'Using geometric features'}")
    print(f"   • PyTorch support: {'Available' if TORCH_AVAILABLE else 'Not available'}")
    print(f"   • Feature extraction: Ready")
    
    print(f"\n💡 Next Steps:")
    print("   1. Prepare diverse training dataset following recommendations")
    print("   2. Extract AU features using this system")
    print("   3. Train multi-modal model combining CNN + AU features")
    print("   4. Validate on diverse test set")

if __name__ == "__main__":
    main()