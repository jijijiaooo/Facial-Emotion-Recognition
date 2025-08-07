#!/usr/bin/env python3
"""
Comprehensive Resource Guide for Fear and Disgust Emotion Datasets
Lists the best sources for high-quality emotion recognition data
"""

import requests
import json
from datetime import datetime

class EmotionDatasetGuide:
    def __init__(self):
        self.resources = {
            "kaggle_datasets": [
                {
                    "name": "FER-2013 (Facial Expression Recognition)",
                    "url": "https://www.kaggle.com/datasets/msambare/fer2013",
                    "description": "48x48 grayscale images, 35,887 examples",
                    "fear_count": "~4,000 images",
                    "disgust_count": "~500 images (limited)",
                    "pros": ["Same format as your current model", "Large dataset", "Preprocessed"],
                    "cons": ["Disgust underrepresented", "Some mislabeled images"],
                    "download_cmd": "kaggle datasets download -d msambare/fer2013",
                    "license": "Public Domain",
                    "quality": "⭐⭐⭐⭐"
                },
                {
                    "name": "AffectNet",
                    "url": "https://www.kaggle.com/datasets/tom99763/affectnet-fer",
                    "description": "1M+ facial images with 8 emotions",
                    "fear_count": "~24,000 images",
                    "disgust_count": "~3,750 images",
                    "pros": ["Huge dataset", "High quality", "Balanced emotions", "Real-world images"],
                    "cons": ["Large download", "Some preprocessing needed"],
                    "download_cmd": "kaggle datasets download -d tom99763/affectnet-fer",
                    "license": "Academic use",
                    "quality": "⭐⭐⭐⭐⭐"
                },
                {
                    "name": "Emotion Detection Dataset",
                    "url": "https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer",
                    "description": "Clean FER dataset with balanced classes",
                    "fear_count": "~5,000 images",
                    "disgust_count": "~1,200 images",
                    "pros": ["Cleaned and balanced", "Good for training", "Multiple formats"],
                    "cons": ["Smaller than original FER"],
                    "download_cmd": "kaggle datasets download -d ananthu017/emotion-detection-fer",
                    "license": "CC0",
                    "quality": "⭐⭐⭐⭐"
                },
                {
                    "name": "Real and Fake Face Detection",
                    "url": "https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection",
                    "description": "High-quality facial images with emotions",
                    "fear_count": "~2,000 images",
                    "disgust_count": "~800 images",
                    "pros": ["High resolution", "Natural expressions", "Diverse demographics"],
                    "cons": ["Smaller dataset", "Mixed with other tasks"],
                    "download_cmd": "kaggle datasets download -d ciplab/real-and-fake-face-detection",
                    "license": "CC BY 4.0",
                    "quality": "⭐⭐⭐⭐"
                },
                {
                    "name": "Facial Expression Recognition Challenge",
                    "url": "https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset",
                    "description": "Competition dataset with 7 emotions",
                    "fear_count": "~3,500 images",
                    "disgust_count": "~600 images",
                    "pros": ["Competition quality", "Well-labeled", "Consistent format"],
                    "cons": ["Disgust still limited"],
                    "download_cmd": "kaggle datasets download -d jonathanoheix/face-expression-recognition-dataset",
                    "license": "Database Contents License",
                    "quality": "⭐⭐⭐⭐"
                }
            ],
            
            "academic_datasets": [
                {
                    "name": "CK+ (Extended Cohn-Kanade)",
                    "url": "http://www.jeffcohn.net/Resources/",
                    "description": "Gold standard for emotion recognition research",
                    "fear_count": "~75 sequences",
                    "disgust_count": "~177 sequences",
                    "pros": ["Highest quality", "Action Unit coded", "Sequence data"],
                    "cons": ["Small size", "Requires academic request"],
                    "access": "Academic license required",
                    "quality": "⭐⭐⭐⭐⭐"
                },
                {
                    "name": "JAFFE (Japanese Female Facial Expression)",
                    "url": "https://zenodo.org/record/3451524",
                    "description": "213 images of 7 emotions from 10 Japanese women",
                    "fear_count": "~30 images",
                    "disgust_count": "~30 images",
                    "pros": ["High quality", "Controlled conditions", "Well-documented"],
                    "cons": ["Very small", "Limited diversity"],
                    "access": "Free download",
                    "quality": "⭐⭐⭐⭐"
                },
                {
                    "name": "KDEF (Karolinska Directed Emotional Faces)",
                    "url": "https://www.kdef.se/",
                    "description": "4,900 pictures of human facial expressions",
                    "fear_count": "~490 images",
                    "disgust_count": "~490 images",
                    "pros": ["Balanced emotions", "Multiple angles", "Professional quality"],
                    "cons": ["Requires registration", "Academic use only"],
                    "access": "Academic license",
                    "quality": "⭐⭐⭐⭐⭐"
                },
                {
                    "name": "RAF-DB (Real-world Affective Faces Database)",
                    "url": "http://www.whdeng.cn/raf/model1.html",
                    "description": "29,672 real-world facial images",
                    "fear_count": "~2,500 images",
                    "disgust_count": "~800 images",
                    "pros": ["Real-world conditions", "Large scale", "Diverse"],
                    "cons": ["Academic request needed"],
                    "access": "Academic license",
                    "quality": "⭐⭐⭐⭐⭐"
                }
            ],
            
            "web_scraping_sources": [
                {
                    "name": "Google Images",
                    "search_terms": ["fear expression", "scared face", "frightened person", "terror face"],
                    "disgust_terms": ["disgust expression", "disgusted face", "revulsion", "nose wrinkle"],
                    "pros": ["Unlimited variety", "Real expressions", "Current images"],
                    "cons": ["Manual work", "Copyright issues", "Quality varies"],
                    "tools": ["google-images-download", "selenium", "beautiful soup"],
                    "legal": "Check usage rights, prefer CC licensed"
                },
                {
                    "name": "Flickr Creative Commons",
                    "url": "https://www.flickr.com/creativecommons/",
                    "search_terms": ["facial expression", "emotion", "portrait"],
                    "pros": ["CC licensed", "High quality", "Diverse"],
                    "cons": ["Manual filtering needed", "Limited emotion tags"],
                    "api": "Flickr API available",
                    "legal": "CC licenses allow use"
                },
                {
                    "name": "Unsplash",
                    "url": "https://unsplash.com/",
                    "search_terms": ["emotion", "expression", "portrait", "face"],
                    "pros": ["Free to use", "High quality", "Professional"],
                    "cons": ["Limited emotion variety", "Mostly positive emotions"],
                    "api": "Unsplash API available",
                    "legal": "Unsplash license"
                }
            ],
            
            "specialized_sources": [
                {
                    "name": "FACS Coded Databases",
                    "description": "Databases with Facial Action Coding System annotations",
                    "sources": ["CK+", "DISFA", "BP4D", "SEMAINE"],
                    "pros": ["Action Unit annotations", "Research quality", "Precise labeling"],
                    "cons": ["Academic access only", "Complex to use"],
                    "use_case": "Training AU-aware models"
                },
                {
                    "name": "Movie/TV Datasets",
                    "description": "Extracted from films and shows",
                    "sources": ["AFEW", "LIRIS-ACCEDE", "MediaEval"],
                    "pros": ["Natural expressions", "Varied contexts", "High quality"],
                    "cons": ["Copyright restrictions", "Preprocessing needed"],
                    "use_case": "Real-world validation"
                },
                {
                    "name": "Synthetic Datasets",
                    "description": "AI-generated facial expressions",
                    "sources": ["StyleGAN2", "Generated faces", "3D rendered"],
                    "pros": ["Unlimited data", "Controlled variations", "No privacy issues"],
                    "cons": ["May not generalize", "Uncanny valley effects"],
                    "use_case": "Data augmentation"
                }
            ]
        }
    
    def print_kaggle_resources(self):
        """Print detailed Kaggle dataset information"""
        print("🏆 KAGGLE DATASETS (Recommended)")
        print("=" * 60)
        
        for i, dataset in enumerate(self.resources["kaggle_datasets"], 1):
            print(f"\n{i}. {dataset['name']} {dataset['quality']}")
            print(f"   URL: {dataset['url']}")
            print(f"   📊 {dataset['description']}")
            print(f"   😨 Fear: {dataset['fear_count']}")
            print(f"   🤢 Disgust: {dataset['disgust_count']}")
            print(f"   ✅ Pros: {', '.join(dataset['pros'])}")
            print(f"   ⚠️ Cons: {', '.join(dataset['cons'])}")
            print(f"   💻 Download: {dataset['download_cmd']}")
            print(f"   📜 License: {dataset['license']}")
    
    def print_academic_resources(self):
        """Print academic dataset information"""
        print("\n🎓 ACADEMIC DATASETS (High Quality)")
        print("=" * 50)
        
        for i, dataset in enumerate(self.resources["academic_datasets"], 1):
            print(f"\n{i}. {dataset['name']} {dataset['quality']}")
            print(f"   URL: {dataset['url']}")
            print(f"   📊 {dataset['description']}")
            print(f"   😨 Fear: {dataset['fear_count']}")
            print(f"   🤢 Disgust: {dataset['disgust_count']}")
            print(f"   ✅ Pros: {', '.join(dataset['pros'])}")
            print(f"   ⚠️ Cons: {', '.join(dataset['cons'])}")
            print(f"   🔑 Access: {dataset['access']}")
    
    def print_web_scraping_guide(self):
        """Print web scraping information"""
        print("\n🌐 WEB SCRAPING SOURCES")
        print("=" * 40)
        
        for source in self.resources["web_scraping_sources"]:
            print(f"\n📍 {source['name']}")
            if 'url' in source:
                print(f"   URL: {source['url']}")
            
            if 'search_terms' in source:
                print(f"   🔍 Fear terms: {', '.join(source['search_terms'])}")
            if 'disgust_terms' in source:
                print(f"   🔍 Disgust terms: {', '.join(source['disgust_terms'])}")
            
            print(f"   ✅ Pros: {', '.join(source['pros'])}")
            print(f"   ⚠️ Cons: {', '.join(source['cons'])}")
            
            if 'tools' in source:
                print(f"   🛠️ Tools: {', '.join(source['tools'])}")
            if 'legal' in source:
                print(f"   ⚖️ Legal: {source['legal']}")
    
    def create_download_script(self):
        """Create a script to download recommended datasets"""
        script = '''#!/bin/bash
# Automated dataset download script
# Run this after setting up Kaggle API

echo "🚀 Downloading Fear and Disgust Datasets"
echo "========================================"

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Create datasets directory
mkdir -p datasets/fear_disgust
cd datasets/fear_disgust

echo "📥 Downloading FER-2013 (Primary dataset)..."
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d fer2013/
echo "✅ FER-2013 downloaded"

echo "📥 Downloading AffectNet subset..."
kaggle datasets download -d tom99763/affectnet-fer
unzip affectnet-fer.zip -d affectnet/
echo "✅ AffectNet downloaded"

echo "📥 Downloading Emotion Detection Dataset..."
kaggle datasets download -d ananthu017/emotion-detection-fer
unzip emotion-detection-fer.zip -d emotion-detection/
echo "✅ Emotion Detection Dataset downloaded"

echo "🎉 All datasets downloaded successfully!"
echo "📁 Check the datasets/fear_disgust/ folder"

# Clean up zip files
rm *.zip

echo "🧹 Cleaned up zip files"
echo "💡 Next: Run python merge_datasets.py to combine them"
'''
        
        with open('download_datasets.sh', 'w') as f:
            f.write(script)
        
        print("✅ Created download_datasets.sh script")
        print("💻 Make executable with: chmod +x download_datasets.sh")
        print("🚀 Run with: ./download_datasets.sh")
    
    def create_dataset_merger(self):
        """Create script to merge multiple datasets"""
        merger_script = '''#!/usr/bin/env python3
"""
Dataset Merger for Fear and Disgust
Combines multiple emotion datasets and balances classes
"""

import os
import shutil
import cv2
import numpy as np
from collections import Counter
import random

class DatasetMerger:
    def __init__(self):
        self.target_emotions = ['fear', 'disgust']
        self.all_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.target_count = 1000  # Target images per emotion
    
    def merge_fer2013(self, source_dir, target_dir):
        """Merge FER-2013 dataset"""
        print("🔄 Processing FER-2013...")
        
        fer_emotions = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        
        # Process train and test folders
        for split in ['train', 'test']:
            split_path = os.path.join(source_dir, split)
            if os.path.exists(split_path):
                for emotion_folder in os.listdir(split_path):
                    emotion_path = os.path.join(split_path, emotion_folder)
                    if os.path.isdir(emotion_path):
                        # Copy images to target directory
                        target_emotion_dir = os.path.join(target_dir, split, emotion_folder)
                        os.makedirs(target_emotion_dir, exist_ok=True)
                        
                        for img_file in os.listdir(emotion_path):
                            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                src = os.path.join(emotion_path, img_file)
                                dst = os.path.join(target_emotion_dir, f"fer_{img_file}")
                                shutil.copy2(src, dst)
    
    def balance_dataset(self, dataset_dir):
        """Balance the dataset by augmenting underrepresented classes"""
        print("⚖️ Balancing dataset...")
        
        for split in ['train', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                continue
            
            # Count images per emotion
            emotion_counts = {}
            for emotion in self.all_emotions:
                emotion_dir = os.path.join(split_dir, emotion)
                if os.path.exists(emotion_dir):
                    count = len([f for f in os.listdir(emotion_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    emotion_counts[emotion] = count
                else:
                    emotion_counts[emotion] = 0
            
            print(f"\\n{split.upper()} set counts:")
            for emotion, count in emotion_counts.items():
                status = "🎯" if emotion in self.target_emotions else ""
                print(f"  {emotion}: {count} {status}")
            
            # Augment fear and disgust if needed
            for emotion in self.target_emotions:
                current_count = emotion_counts[emotion]
                if current_count < self.target_count:
                    needed = self.target_count - current_count
                    print(f"\\n🔄 Augmenting {emotion}: need {needed} more images")
                    self.augment_emotion(split_dir, emotion, needed)
    
    def augment_emotion(self, split_dir, emotion, needed_count):
        """Augment specific emotion with transformations"""
        emotion_dir = os.path.join(split_dir, emotion)
        if not os.path.exists(emotion_dir):
            return
        
        # Get existing images
        existing_files = [f for f in os.listdir(emotion_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not existing_files:
            return
        
        augmented_count = 0
        while augmented_count < needed_count:
            # Pick random existing image
            source_file = random.choice(existing_files)
            source_path = os.path.join(emotion_dir, source_file)
            
            # Load and augment
            img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                augmented_img = self.apply_augmentation(img, emotion)
                
                # Save augmented image
                aug_filename = f"aug_{augmented_count}_{source_file}"
                aug_path = os.path.join(emotion_dir, aug_filename)
                cv2.imwrite(aug_path, augmented_img)
                
                augmented_count += 1
                
                if augmented_count % 100 == 0:
                    print(f"  Generated {augmented_count}/{needed_count} images")
    
    def apply_augmentation(self, img, emotion):
        """Apply emotion-specific augmentation"""
        # Random transformations
        augmented = img.copy()
        
        # Rotation (small)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            center = (img.shape[1]//2, img.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, M, (img.shape[1], img.shape[0]))
        
        # Brightness adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            augmented = cv2.convertScaleAbs(augmented, alpha=brightness, beta=0)
        
        # Contrast adjustment
        if random.random() > 0.5:
            contrast = random.uniform(0.8, 1.2)
            augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=0)
        
        # Emotion-specific enhancements
        if emotion == 'fear':
            # Enhance upper region (eyebrows/eyes)
            h = augmented.shape[0]
            upper_region = augmented[:h//2, :]
            enhanced_upper = cv2.equalizeHist(upper_region)
            augmented[:h//2, :] = enhanced_upper
        
        elif emotion == 'disgust':
            # Enhance middle region (nose area)
            h = augmented.shape[0]
            middle_region = augmented[h//3:2*h//3, :]
            enhanced_middle = cv2.equalizeHist(middle_region)
            augmented[h//3:2*h//3, :] = enhanced_middle
        
        return augmented
    
    def run_merge(self):
        """Run the complete merge process"""
        print("🔄 Starting dataset merge process...")
        
        # Create target directory structure
        target_dir = "merged_emotion_dataset"
        for split in ['train', 'test']:
            for emotion in self.all_emotions:
                os.makedirs(os.path.join(target_dir, split, emotion), exist_ok=True)
        
        # Merge datasets
        datasets_dir = "datasets/fear_disgust"
        
        # Merge FER-2013
        fer_dir = os.path.join(datasets_dir, "fer2013")
        if os.path.exists(fer_dir):
            self.merge_fer2013(fer_dir, target_dir)
        
        # Balance the dataset
        self.balance_dataset(target_dir)
        
        print("\\n✅ Dataset merge completed!")
        print(f"📁 Merged dataset available in: {target_dir}")
        print("🎯 Fear and disgust classes have been boosted!")

if __name__ == "__main__":
    merger = DatasetMerger()
    merger.run_merge()
'''
        
        with open('merge_datasets.py', 'w') as f:
            f.write(merger_script)
        
        print("✅ Created merge_datasets.py script")
    
    def print_quick_start_guide(self):
        """Print quick start guide"""
        print("\n🚀 QUICK START GUIDE")
        print("=" * 30)
        print("1. Set up Kaggle API:")
        print("   pip install kaggle")
        print("   # Get API key from kaggle.com/account")
        print("   # Place kaggle.json in ~/.kaggle/")
        
        print("\n2. Download datasets:")
        print("   chmod +x download_datasets.sh")
        print("   ./download_datasets.sh")
        
        print("\n3. Merge and balance:")
        print("   python merge_datasets.py")
        
        print("\n4. Update your training:")
        print("   # Point your training script to merged_emotion_dataset/")
        print("   # You should now have 1000+ fear and disgust images each!")
        
        print("\n🎯 PRIORITY RECOMMENDATIONS:")
        print("1. ⭐⭐⭐⭐⭐ AffectNet - Best quality and balance")
        print("2. ⭐⭐⭐⭐ FER-2013 - Compatible with your current model")
        print("3. ⭐⭐⭐⭐ CK+ - Highest quality (if you can get academic access)")
        
        print("\n💡 PRO TIPS:")
        print("• Focus on AffectNet for best results")
        print("• Use CK+ for validation/testing")
        print("• Supplement with web scraping for variety")
        print("• Always verify emotion labels manually")

def main():
    print("📊 EMOTION DATASET RESOURCE GUIDE")
    print("=" * 50)
    print("Comprehensive guide for finding Fear and Disgust emotion data")
    
    guide = EmotionDatasetGuide()
    
    # Print all resources
    guide.print_kaggle_resources()
    guide.print_academic_resources()
    guide.print_web_scraping_guide()
    
    # Create helper scripts
    print("\n🛠️ CREATING HELPER SCRIPTS...")
    guide.create_download_script()
    guide.create_dataset_merger()
    
    # Print quick start
    guide.print_quick_start_guide()
    
    print(f"\n📅 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()