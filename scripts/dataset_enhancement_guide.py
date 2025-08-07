#!/usr/bin/env python3
"""
Dataset Enhancement Guide for Better Emotion Recognition
Practical steps to improve your training data with Action Units and diversity
"""

import cv2
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

class DatasetAnalyzer:
    """Analyze and enhance your emotion recognition dataset"""
    
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Critical Action Units for each emotion (based on FACS research)
        self.critical_aus = {
            'happy': {
                'primary': [6, 12],      # Cheek raiser + Lip corner puller
                'secondary': [25, 26],   # Lips part + Jaw drop (for laughter)
                'description': 'Genuine smile with eye crinkles'
            },
            'sad': {
                'primary': [1, 4, 15],   # Inner brow raiser + Brow lowerer + Lip corner depressor
                'secondary': [17, 11],   # Chin raiser + Nasolabial deepener
                'description': 'Drooping features, inner brow raise'
            },
            'angry': {
                'primary': [4, 5, 7, 23], # Brow lowerer + Upper lid raiser + Lid tightener + Lip tightener
                'secondary': [10, 22, 24], # Upper lip raiser + Lip funneler + Lip pressor
                'description': 'Furrowed brow, tense eyes and mouth'
            },
            'fear': {
                'primary': [1, 2, 4, 5, 20], # Inner+Outer brow raiser + Brow lowerer + Upper lid raiser + Lip stretcher
                'secondary': [25, 26, 27],    # Lips part + Jaw drop + Mouth stretch
                'description': 'Wide eyes, raised eyebrows, open mouth'
            },
            'surprise': {
                'primary': [1, 2, 5, 26], # Inner+Outer brow raiser + Upper lid raiser + Jaw drop
                'secondary': [25, 27],     # Lips part + Mouth stretch
                'description': 'Raised eyebrows, wide eyes, dropped jaw'
            },
            'disgust': {
                'primary': [9, 15, 16],   # Nose wrinkler + Lip corner depressor + Lower lip depressor
                'secondary': [4, 7, 10],  # Brow lowerer + Lid tightener + Upper lip raiser
                'description': 'Wrinkled nose, raised upper lip'
            },
            'neutral': {
                'primary': [],
                'secondary': [],
                'description': 'Relaxed facial muscles, no active AUs'
            }
        }
    
    def analyze_dataset_balance(self):
        """Analyze class distribution in your dataset"""
        print("📊 Dataset Balance Analysis")
        print("-" * 40)
        
        emotion_counts = {}
        total_images = 0
        
        for emotion in self.emotions:
            emotion_path = os.path.join(self.data_path, "train", emotion)
            if os.path.exists(emotion_path):
                count = len([f for f in os.listdir(emotion_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                emotion_counts[emotion] = count
                total_images += count
                print(f"{emotion.capitalize():10}: {count:5d} images")
            else:
                emotion_counts[emotion] = 0
                print(f"{emotion.capitalize():10}: {0:5d} images (folder not found)")
        
        print(f"{'Total':10}: {total_images:5d} images")
        
        # Check balance
        if total_images > 0:
            print(f"\n📈 Balance Analysis:")
            avg_per_class = total_images / len(self.emotions)
            for emotion, count in emotion_counts.items():
                percentage = (count / total_images) * 100
                balance_status = "✅ Good" if abs(count - avg_per_class) < avg_per_class * 0.2 else "⚠️ Imbalanced"
                print(f"{emotion.capitalize():10}: {percentage:5.1f}% {balance_status}")
        
        return emotion_counts
    
    def check_image_quality(self, sample_size=50):
        """Check image quality and diversity"""
        print(f"\n🔍 Image Quality Analysis (sampling {sample_size} images per emotion)")
        print("-" * 60)
        
        quality_stats = {}
        
        for emotion in self.emotions:
            emotion_path = os.path.join(self.data_path, "train", emotion)
            if not os.path.exists(emotion_path):
                continue
            
            image_files = [f for f in os.listdir(emotion_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                continue
            
            # Sample images
            sample_files = np.random.choice(image_files, 
                                          min(sample_size, len(image_files)), 
                                          replace=False)
            
            sizes = []
            brightnesses = []
            contrasts = []
            
            for img_file in sample_files:
                img_path = os.path.join(emotion_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    sizes.append(img.shape)
                    brightnesses.append(np.mean(img))
                    contrasts.append(np.std(img))
            
            if sizes:
                unique_sizes = len(set(sizes))
                avg_brightness = np.mean(brightnesses)
                avg_contrast = np.mean(contrasts)
                
                quality_stats[emotion] = {
                    'unique_sizes': unique_sizes,
                    'avg_brightness': avg_brightness,
                    'avg_contrast': avg_contrast,
                    'brightness_std': np.std(brightnesses),
                    'contrast_std': np.std(contrasts)
                }
                
                print(f"{emotion.capitalize():10}:")
                print(f"  Size variety: {unique_sizes} different sizes")
                print(f"  Brightness:   {avg_brightness:.1f} ± {np.std(brightnesses):.1f}")
                print(f"  Contrast:     {avg_contrast:.1f} ± {np.std(contrasts):.1f}")
                
                # Quality recommendations
                recommendations = []
                if unique_sizes < 3:
                    recommendations.append("Add more size variety")
                if avg_brightness < 80:
                    recommendations.append("Add brighter images")
                elif avg_brightness > 180:
                    recommendations.append("Add darker images")
                if avg_contrast < 30:
                    recommendations.append("Add higher contrast images")
                if np.std(brightnesses) < 20:
                    recommendations.append("Increase lighting diversity")
                
                if recommendations:
                    print(f"  💡 Suggestions: {', '.join(recommendations)}")
                else:
                    print(f"  ✅ Good diversity")
        
        return quality_stats
    
    def generate_enhancement_plan(self, emotion_counts):
        """Generate specific enhancement recommendations"""
        print(f"\n🎯 Dataset Enhancement Plan")
        print("=" * 50)
        
        total_images = sum(emotion_counts.values())
        if total_images == 0:
            print("❌ No images found in dataset")
            return
        
        target_per_class = max(emotion_counts.values())  # Target the largest class
        
        print(f"Current total: {total_images} images")
        print(f"Target per class: {target_per_class} images")
        print(f"Target total: {target_per_class * len(self.emotions)} images")
        
        print(f"\n📋 Specific Actions Needed:")
        print("-" * 30)
        
        for emotion in self.emotions:
            current_count = emotion_counts.get(emotion, 0)
            needed = target_per_class - current_count
            
            if needed > 0:
                print(f"\n{emotion.upper()}:")
                print(f"  Current: {current_count}, Need: {needed} more images")
                
                # Specific AU recommendations
                au_info = self.critical_aus[emotion]
                print(f"  🎭 Focus on Action Units: {au_info['primary']}")
                print(f"  📝 Description: {au_info['description']}")
                
                # Diversity recommendations
                print(f"  🌟 Collect diverse examples:")
                print(f"    • Different ages (children, adults, elderly)")
                print(f"    • Various intensities (subtle to strong)")
                print(f"    • Different lighting conditions")
                print(f"    • Multiple ethnicities and genders")
                
                if emotion == 'happy':
                    print(f"    • Include both posed and genuine smiles")
                    print(f"    • Duchenne smiles (with eye crinkles)")
                elif emotion == 'angry':
                    print(f"    • Include mild irritation to rage")
                    print(f"    • Both symmetric and asymmetric expressions")
                elif emotion == 'sad':
                    print(f"    • From melancholy to crying")
                    print(f"    • Include teary eyes variations")
        
        # Data augmentation recommendations
        print(f"\n🔄 Data Augmentation Strategy:")
        print("-" * 35)
        print("For each emotion, apply these augmentations:")
        print("  • Rotation: ±10 degrees")
        print("  • Brightness: ±15%")
        print("  • Contrast: ±15%")
        print("  • Horizontal flip (careful with asymmetric expressions)")
        print("  • Slight zoom: 0.95-1.05x")
        print("  • Add subtle noise")
        
        # Quality control
        print(f"\n✅ Quality Control Checklist:")
        print("-" * 30)
        print("  □ Remove blurry or low-quality images")
        print("  □ Verify emotion labels are correct")
        print("  □ Remove duplicate or near-duplicate images")
        print("  □ Ensure faces are properly cropped")
        print("  □ Check for consistent image sizes")
        print("  □ Remove images with multiple faces")
        print("  □ Validate lighting is adequate")
    
    def create_collection_guidelines(self):
        """Create guidelines for collecting new data"""
        print(f"\n📸 Data Collection Guidelines")
        print("=" * 40)
        
        guidelines = {
            "Technical Requirements": [
                "Minimum resolution: 48x48 pixels (preferably higher)",
                "Grayscale or color (will be converted to grayscale)",
                "Clear, unblurred images",
                "Single face per image",
                "Face should occupy 60-80% of image"
            ],
            
            "Diversity Requirements": [
                "Age range: 5-80 years",
                "Gender balance: 50/50 male/female",
                "Multiple ethnicities represented",
                "Various lighting conditions",
                "Different backgrounds (but face should be prominent)"
            ],
            
            "Expression Guidelines": [
                "Natural, spontaneous expressions preferred",
                "Include micro-expressions (subtle)",
                "Include macro-expressions (obvious)",
                "Capture transition moments",
                "Avoid overly posed expressions"
            ],
            
            "Action Unit Focus": [
                "Study FACS (Facial Action Coding System)",
                "Ensure critical AUs are present for each emotion",
                "Include AU combinations",
                "Document which AUs are visible in each image"
            ]
        }
        
        for category, items in guidelines.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
        
        # Emotion-specific collection tips
        print(f"\n🎭 Emotion-Specific Collection Tips:")
        print("-" * 40)
        
        tips = {
            'happy': "Look for genuine Duchenne smiles with eye crinkles, not just mouth smiles",
            'sad': "Capture the drooping of facial features, especially mouth corners and inner eyebrows",
            'angry': "Focus on furrowed brows and tense jaw/mouth area",
            'fear': "Wide eyes and raised eyebrows are key, often with open mouth",
            'surprise': "Similar to fear but more symmetrical, with dropped jaw",
            'disgust': "Wrinkled nose and raised upper lip are characteristic",
            'neutral': "Relaxed face with no active muscle tension"
        }
        
        for emotion, tip in tips.items():
            print(f"{emotion.capitalize():10}: {tip}")

def main():
    print("🎭 Dataset Enhancement Guide for Emotion Recognition")
    print("=" * 60)
    print("This tool helps you improve your training dataset for better accuracy")
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer()
    
    # Analyze current dataset
    emotion_counts = analyzer.analyze_dataset_balance()
    
    # Check image quality
    analyzer.check_image_quality()
    
    # Generate enhancement plan
    analyzer.generate_enhancement_plan(emotion_counts)
    
    # Provide collection guidelines
    analyzer.create_collection_guidelines()
    
    print(f"\n🚀 Summary:")
    print("=" * 20)
    print("1. ✅ Diverse Action Units improve accuracy by 15-25%")
    print("2. ✅ Balanced datasets prevent bias")
    print("3. ✅ Quality control ensures reliable training")
    print("4. ✅ Follow FACS guidelines for authentic expressions")
    print("5. ✅ Use data augmentation to increase variety")
    
    print(f"\n📚 Recommended Reading:")
    print("• Facial Action Coding System (FACS) by Ekman & Friesen")
    print("• 'Emotion Recognition in the Wild' research papers")
    print("• AffectNet and FER2013 dataset papers for best practices")

if __name__ == "__main__":
    main()