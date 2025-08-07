#!/usr/bin/env python3
"""
Fear and Disgust Detection Diagnostic Tool
Analyzes why these emotions are not being recognized properly
"""

import cv2
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

class FearDisgustDiagnostic:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Critical visual features for fear and disgust
        self.fear_features = {
            'eyes': 'Wide open, showing white above iris',
            'eyebrows': 'Raised high, creating forehead wrinkles',
            'mouth': 'Open, lips stretched horizontally',
            'overall': 'Symmetrical expression of alarm'
        }
        
        self.disgust_features = {
            'nose': 'Wrinkled, nostrils may flare',
            'upper_lip': 'Raised, showing upper teeth',
            'mouth': 'Corners turned down',
            'eyes': 'May be squinted or narrowed'
        }
        
        # Common misclassifications
        self.common_mistakes = {
            'fear_as_surprise': 'Both have raised eyebrows and wide eyes',
            'fear_as_neutral': 'Subtle fear expressions look neutral',
            'disgust_as_angry': 'Both can have furrowed brows',
            'disgust_as_sad': 'Both have downturned mouth corners'
        }
    
    def analyze_class_distribution(self):
        """Analyze the distribution focusing on fear and disgust"""
        print("🔍 Fear and Disgust Class Analysis")
        print("=" * 50)
        
        emotion_counts = {}
        sample_paths = {}
        
        for emotion in self.emotions:
            train_path = os.path.join(self.data_path, "train", emotion)
            test_path = os.path.join(self.data_path, "test", emotion)
            
            train_count = 0
            test_count = 0
            
            if os.path.exists(train_path):
                train_files = [f for f in os.listdir(train_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                train_count = len(train_files)
                if train_files:
                    sample_paths[f"{emotion}_train"] = os.path.join(train_path, train_files[0])
            
            if os.path.exists(test_path):
                test_files = [f for f in os.listdir(test_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                test_count = len(test_files)
                if test_files:
                    sample_paths[f"{emotion}_test"] = os.path.join(test_path, test_files[0])
            
            emotion_counts[emotion] = {'train': train_count, 'test': test_count}
            total = train_count + test_count
            
            status = "✅" if total > 500 else "⚠️" if total > 100 else "❌"
            print(f"{emotion.capitalize():10}: Train={train_count:4d}, Test={test_count:4d}, Total={total:4d} {status}")
        
        # Focus on fear and disgust
        print(f"\n🎯 Fear and Disgust Detailed Analysis:")
        print("-" * 40)
        
        fear_total = emotion_counts['fear']['train'] + emotion_counts['fear']['test']
        disgust_total = emotion_counts['disgust']['train'] + emotion_counts['disgust']['test']
        
        print(f"Fear total: {fear_total} images")
        if fear_total < 300:
            print("❌ CRITICAL: Fear has too few examples (need 500+ minimum)")
        elif fear_total < 500:
            print("⚠️ WARNING: Fear is underrepresented (recommend 1000+)")
        else:
            print("✅ Fear has adequate representation")
        
        print(f"Disgust total: {disgust_total} images")
        if disgust_total < 300:
            print("❌ CRITICAL: Disgust has too few examples (need 500+ minimum)")
        elif disgust_total < 500:
            print("⚠️ WARNING: Disgust is underrepresented (recommend 1000+)")
        else:
            print("✅ Disgust has adequate representation")
        
        return emotion_counts, sample_paths
    
    def analyze_image_quality(self, sample_paths):
        """Analyze the quality of fear and disgust images"""
        print(f"\n🔬 Image Quality Analysis for Fear and Disgust")
        print("-" * 50)
        
        target_emotions = ['fear', 'disgust']
        
        for emotion in target_emotions:
            print(f"\n{emotion.upper()} Analysis:")
            print("-" * 20)
            
            # Analyze training samples
            train_key = f"{emotion}_train"
            if train_key in sample_paths:
                img_path = sample_paths[train_key]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Basic quality metrics
                    brightness = np.mean(img)
                    contrast = np.std(img)
                    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
                    
                    print(f"Sample image: {os.path.basename(img_path)}")
                    print(f"Brightness: {brightness:.1f} (good: 80-180)")
                    print(f"Contrast: {contrast:.1f} (good: >30)")
                    print(f"Sharpness: {sharpness:.1f} (good: >100)")
                    
                    # Quality assessment
                    issues = []
                    if brightness < 60:
                        issues.append("Too dark")
                    elif brightness > 200:
                        issues.append("Too bright")
                    if contrast < 25:
                        issues.append("Low contrast")
                    if sharpness < 50:
                        issues.append("Blurry")
                    
                    if issues:
                        print(f"⚠️ Issues found: {', '.join(issues)}")
                    else:
                        print("✅ Good image quality")
                    
                    # Analyze facial features specific to this emotion
                    self.analyze_facial_features(img, emotion)
                else:
                    print("❌ Could not load sample image")
            else:
                print("❌ No sample images found")
    
    def analyze_facial_features(self, img, emotion):
        """Analyze if the image contains the right facial features for the emotion"""
        print(f"Facial feature analysis:")
        
        h, w = img.shape
        
        # Divide face into regions
        top_third = img[:h//3, :]          # Forehead/eyebrows
        middle_third = img[h//3:2*h//3, :] # Eyes/nose
        bottom_third = img[2*h//3:, :]     # Mouth/chin
        
        # Analyze brightness patterns
        top_brightness = np.mean(top_third)
        middle_brightness = np.mean(middle_third)
        bottom_brightness = np.mean(bottom_third)
        
        if emotion == 'fear':
            # Fear typically has:
            # - Bright forehead (raised eyebrows create wrinkles)
            # - Wide eyes (more white showing)
            # - Open mouth (darker bottom region)
            
            forehead_raised = top_brightness > middle_brightness * 1.1
            eyes_wide = middle_brightness > np.mean(img) * 1.05
            mouth_open = bottom_brightness < np.mean(img) * 0.95
            
            print(f"  Raised eyebrows: {'✅' if forehead_raised else '❌'}")
            print(f"  Wide eyes: {'✅' if eyes_wide else '❌'}")
            print(f"  Open mouth: {'✅' if mouth_open else '❌'}")
            
            score = sum([forehead_raised, eyes_wide, mouth_open])
            print(f"  Fear feature score: {score}/3")
            
        elif emotion == 'disgust':
            # Disgust typically has:
            # - Wrinkled nose (texture in middle region)
            # - Raised upper lip (specific mouth pattern)
            # - Possible squinted eyes
            
            nose_wrinkled = np.std(middle_third) > np.std(img) * 1.1
            mouth_pattern = np.std(bottom_third) > np.std(img) * 1.05
            
            print(f"  Wrinkled nose: {'✅' if nose_wrinkled else '❌'}")
            print(f"  Mouth expression: {'✅' if mouth_pattern else '❌'}")
            
            score = sum([nose_wrinkled, mouth_pattern])
            print(f"  Disgust feature score: {score}/2")
    
    def test_model_predictions(self):
        """Test current model on fear and disgust samples"""
        print(f"\n🧠 Model Prediction Test")
        print("-" * 30)
        
        try:
            from tensorflow import keras
            model = keras.models.load_model('model_file_30epochs.h5')
            print("✅ Model loaded successfully")
            
            # Test on sample images
            for emotion in ['fear', 'disgust']:
                emotion_path = os.path.join(self.data_path, "train", emotion)
                if os.path.exists(emotion_path):
                    image_files = [f for f in os.listdir(emotion_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    if image_files:
                        # Test first 5 images
                        test_files = image_files[:5]
                        predictions = []
                        
                        for img_file in test_files:
                            img_path = os.path.join(emotion_path, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            
                            if img is not None:
                                # Preprocess like in test.py
                                resized = cv2.resize(img, (48, 48))
                                normalized = resized / 255.0
                                reshaped = np.reshape(normalized, (1, 48, 48, 1))
                                
                                # Predict
                                result = model.predict(reshaped, verbose=0)
                                predicted_idx = np.argmax(result, axis=1)[0]
                                confidence = result[0][predicted_idx]
                                predicted_emotion = self.emotions[predicted_idx]
                                
                                predictions.append({
                                    'file': img_file,
                                    'predicted': predicted_emotion,
                                    'confidence': confidence,
                                    'correct': predicted_emotion == emotion
                                })
                        
                        # Analyze predictions
                        correct_count = sum(1 for p in predictions if p['correct'])
                        accuracy = correct_count / len(predictions) * 100
                        
                        print(f"\n{emotion.upper()} Prediction Results:")
                        print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(predictions)})")
                        
                        # Show misclassifications
                        wrong_predictions = [p for p in predictions if not p['correct']]
                        if wrong_predictions:
                            print("Misclassifications:")
                            for p in wrong_predictions:
                                print(f"  {p['file']}: predicted as {p['predicted']} (conf: {p['confidence']:.2f})")
                        
                        # Show prediction distribution
                        pred_counts = Counter(p['predicted'] for p in predictions)
                        print("Prediction distribution:")
                        for pred_emotion, count in pred_counts.items():
                            print(f"  {pred_emotion}: {count}")
                    
        except Exception as e:
            print(f"❌ Could not test model: {e}")
    
    def generate_improvement_plan(self):
        """Generate specific improvement plan for fear and disgust"""
        print(f"\n🎯 Fear and Disgust Improvement Plan")
        print("=" * 50)
        
        print(f"\n1. DATA COLLECTION PRIORITIES:")
        print("-" * 35)
        print("FEAR - Collect images with:")
        print("  • Wide open eyes (showing white above iris)")
        print("  • Raised eyebrows creating forehead wrinkles")
        print("  • Open mouth in 'O' shape or horizontal stretch")
        print("  • Symmetrical expression (both sides of face)")
        print("  • Various intensities: mild concern → terror")
        
        print("\nDISGUST - Collect images with:")
        print("  • Wrinkled nose (most important feature)")
        print("  • Raised upper lip showing teeth")
        print("  • Downturned mouth corners")
        print("  • Possible squinted eyes")
        print("  • Various triggers: bad smell, taste, sight")
        
        print(f"\n2. AVOID COMMON CONFUSIONS:")
        print("-" * 30)
        print("Fear vs Surprise:")
        print("  • Fear: asymmetrical, mouth open differently")
        print("  • Surprise: more symmetrical, jaw drops straight down")
        
        print("Disgust vs Anger:")
        print("  • Disgust: nose wrinkle is key differentiator")
        print("  • Anger: no nose wrinkle, more jaw tension")
        
        print(f"\n3. TECHNICAL IMPROVEMENTS:")
        print("-" * 25)
        print("  • Increase fear/disgust samples to 1000+ each")
        print("  • Use data augmentation carefully (preserve key features)")
        print("  • Add class weights in training (boost underrepresented classes)")
        print("  • Use focal loss to handle class imbalance")
        print("  • Implement hard negative mining")
        
        print(f"\n4. IMMEDIATE ACTIONS:")
        print("-" * 20)
        print("  □ Run dataset analysis to confirm class imbalance")
        print("  □ Collect 500+ new fear images with proper features")
        print("  □ Collect 500+ new disgust images with nose wrinkles")
        print("  □ Retrain model with balanced dataset")
        print("  □ Test specifically on fear/disgust validation set")
    
    def create_collection_templates(self):
        """Create templates for collecting good fear and disgust images"""
        print(f"\n📸 Collection Templates")
        print("-" * 25)
        
        fear_template = """
FEAR IMAGE COLLECTION CHECKLIST:
□ Eyes: Wide open, showing white above/below iris
□ Eyebrows: Raised high, creating horizontal forehead lines
□ Mouth: Open (oval or horizontal stretch)
□ Expression: Symmetrical on both sides
□ Context: Reaction to threat, surprise, or startling event
□ Intensity: Range from mild concern to extreme terror
□ Quality: Clear, well-lit, 48x48+ resolution
□ Pose: Frontal view, face fills 60-80% of frame
"""
        
        disgust_template = """
DISGUST IMAGE COLLECTION CHECKLIST:
□ Nose: Wrinkled/scrunched (MOST IMPORTANT)
□ Upper lip: Raised, may show upper teeth
□ Mouth: Corners turned down
□ Eyes: May be squinted or normal
□ Expression: Can be asymmetrical
□ Context: Reaction to bad smell, taste, or sight
□ Intensity: Range from mild distaste to extreme revulsion
□ Quality: Clear, well-lit, 48x48+ resolution
□ Pose: Frontal view, face fills 60-80% of frame
"""
        
        print("FEAR Collection Template:")
        print(fear_template)
        print("\nDISGUST Collection Template:")
        print(disgust_template)

def main():
    print("🔍 Fear and Disgust Detection Diagnostic")
    print("=" * 50)
    print("Analyzing why fear and disgust emotions are not being detected...")
    
    diagnostic = FearDisgustDiagnostic()
    
    # Run comprehensive analysis
    emotion_counts, sample_paths = diagnostic.analyze_class_distribution()
    diagnostic.analyze_image_quality(sample_paths)
    diagnostic.test_model_predictions()
    diagnostic.generate_improvement_plan()
    diagnostic.create_collection_templates()
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 20)
    print("Most likely causes of missing fear/disgust detection:")
    print("1. ❌ Insufficient training data for these emotions")
    print("2. ❌ Poor quality images missing key facial features")
    print("3. ❌ Model confusion with similar emotions")
    print("4. ❌ Class imbalance in training dataset")
    
    print(f"\n💡 QUICK FIX:")
    print("Collect 500+ high-quality images each for fear and disgust")
    print("Focus on the key features listed above!")

if __name__ == "__main__":
    main()