#!/usr/bin/env python3
"""
Emotion Detection System Evaluation Script
Evaluates performance using precision, recall, F1-score, and confusion matrix
"""

import os
import sys
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score
)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))
from simple_emotion_detection import SimpleEmotionDetector

class EmotionSystemEvaluator:
    def __init__(self, test_data_path, output_dir='evaluation_results', 
                 sample_size=None, min_samples_per_class=50, max_samples_per_class=200):
        """Initialize evaluator with test data path and output directory
        
        Args:
            test_data_path: Path to test data directory
            output_dir: Output directory for results
            sample_size: Total number of samples to evaluate (None = all samples)
            min_samples_per_class: Minimum samples per emotion class
            max_samples_per_class: Maximum samples per emotion class (for speed)
        """
        self.test_data_path = test_data_path
        self.output_dir = output_dir
        
        # Initialize detector with optimizations
        print("Initializing emotion detector...")
        self.detector = SimpleEmotionDetector()
        self.detector.debug_mode = False  # Disable debug output for speed
        
        # Sampling parameters
        self.sample_size = sample_size
        self.min_samples_per_class = min_samples_per_class
        self.max_samples_per_class = max_samples_per_class
        
        # Emotion labels - ensure order matches the detector
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(output_dir, f'evaluation_{timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Evaluation results will be saved to: {self.results_dir}")
    
    def load_test_images(self):
        """Load test images with stratified sampling for faster, accurate evaluation"""
        print("Loading test images...")
        images = []
        labels = []
        
        # First pass: count all available images
        emotion_files = {}
        total_available = 0
        
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(self.test_data_path, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Warning: Directory not found: {emotion_dir}")
                emotion_files[emotion] = []
                continue
            
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            emotion_files[emotion] = image_files
            total_available += len(image_files)
            print(f"  Found {len(image_files)} images for {emotion}")
        
        print(f"\nTotal available images: {total_available}")
        
        # Determine sampling strategy
        if self.sample_size is not None and self.sample_size < total_available:
            # Stratified sampling - ensure each class is represented proportionally
            print(f"Using stratified sampling: {self.sample_size} samples")
            samples_per_class = max(
                self.min_samples_per_class,
                min(self.max_samples_per_class, self.sample_size // len(self.emotions))
            )
            print(f"Target samples per class: {samples_per_class}")
        else:
            # Use all available, but cap per class for speed
            samples_per_class = self.max_samples_per_class
            print(f"Using all images (max {samples_per_class} per class)")
        
        # Load images with sampling
        np.random.seed(42)  # For reproducible sampling
        
        for emotion_idx, emotion in enumerate(self.emotions):
            files = emotion_files.get(emotion, [])
            if not files:
                continue
            
            # Sample or use all files
            if len(files) > samples_per_class:
                # Randomly sample
                selected_files = np.random.choice(files, samples_per_class, replace=False)
                print(f"  Sampling {samples_per_class}/{len(files)} images for {emotion}")
            else:
                selected_files = files
                print(f"  Using all {len(files)} images for {emotion}")
            
            emotion_dir = os.path.join(self.test_data_path, emotion)
            for img_file in selected_files:
                img_path = os.path.join(emotion_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append((img_path, img))
                        labels.append(emotion_idx)
                except Exception as e:
                    print(f"    Error loading {img_file}: {e}")
        
        print(f"\nTotal images loaded for evaluation: {len(images)}")
        print(f"This is {len(images)/total_available*100:.1f}% of available data\n")
        return images, labels
    
    def evaluate(self):
        """Run evaluation on test dataset"""
        print("\n" + "="*60)
        print("EMOTION DETECTION SYSTEM EVALUATION")
        print("="*60 + "\n")
        
        # Load test data
        images, true_labels = self.load_test_images()
        
        if len(images) == 0:
            print("Error: No test images found!")
            return
        
        # Run predictions
        print("\nRunning predictions...")
        predicted_labels = []
        confidences = []
        failed_predictions = []
        
        # Batch processing for efficiency
        batch_size = 50
        total_batches = (len(images) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(images))
            batch = images[start_idx:end_idx]
            
            print(f"  Batch {batch_idx + 1}/{total_batches}: Processing images {start_idx}-{end_idx-1}...")
            
            for img_path, img in batch:
                try:
                    # Detect emotion
                    emotion, confidence = self.detector.predict_emotion(img)
                    
                    # Convert emotion string to index
                    emotion_lower = emotion.lower()
                    if emotion_lower in self.emotions:
                        pred_idx = self.emotions.index(emotion_lower)
                        predicted_labels.append(pred_idx)
                        confidences.append(confidence)
                    else:
                        # Unknown emotion - default to neutral
                        predicted_labels.append(4)  # neutral
                        confidences.append(0.0)
                        failed_predictions.append(img_path)
                except Exception as e:
                    print(f"    Error predicting {img_path}: {e}")
                    predicted_labels.append(4)  # default to neutral
                    confidences.append(0.0)
                    failed_predictions.append(img_path)
        
        print(f"\n✓ Predictions complete! Failed: {len(failed_predictions)}/{len(images)}")
        
        # Calculate metrics
        print("\n" + "="*60)
        print("CALCULATING PERFORMANCE METRICS")
        print("="*60 + "\n")
        
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        # Overall accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, 
            predicted_labels, 
            labels=range(len(self.emotions)),
            zero_division=0
        )
        
        # Print detailed metrics
        print("\nPer-Emotion Metrics:")
        print("-" * 80)
        print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        
        metrics_data = []
        for i, emotion in enumerate(self.emotions):
            print(f"{emotion.capitalize():<12} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")
            metrics_data.append({
                'emotion': emotion,
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            })
        
        print("-" * 80)
        print(f"{'Macro Avg':<12} {np.mean(precision):<12.4f} {np.mean(recall):<12.4f} "
              f"{np.mean(f1):<12.4f} {np.sum(support):<10}")
        print(f"{'Weighted Avg':<12} "
              f"{np.average(precision, weights=support):<12.4f} "
              f"{np.average(recall, weights=support):<12.4f} "
              f"{np.average(f1, weights=support):<12.4f} "
              f"{np.sum(support):<10}")
        print("-" * 80)
        
        # Generate classification report
        report = classification_report(
            true_labels, 
            predicted_labels,
            target_names=[e.capitalize() for e in self.emotions],
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        # Save results
        self.save_results(
            accuracy=accuracy,
            metrics_data=metrics_data,
            classification_report=report,
            confusion_matrix=cm,
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            confidences=confidences,
            failed_predictions=failed_predictions
        )
        
        # Generate visualizations
        self.generate_visualizations(cm, metrics_data)
        
        print(f"\n✓ Evaluation complete! Results saved to: {self.results_dir}")
        return accuracy, metrics_data, cm
    
    def save_results(self, accuracy, metrics_data, classification_report, 
                     confusion_matrix, true_labels, predicted_labels, 
                     confidences, failed_predictions):
        """Save evaluation results to files"""
        print("\nSaving results...")
        
        # Save JSON summary
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'test_data_path': self.test_data_path,
            'total_samples': len(true_labels),
            'overall_accuracy': float(accuracy),
            'failed_predictions': len(failed_predictions),
            'metrics_per_emotion': metrics_data,
            'macro_avg': {
                'precision': float(np.mean([m['precision'] for m in metrics_data])),
                'recall': float(np.mean([m['recall'] for m in metrics_data])),
                'f1_score': float(np.mean([m['f1_score'] for m in metrics_data]))
            },
            'weighted_avg': {
                'precision': float(np.average(
                    [m['precision'] for m in metrics_data],
                    weights=[m['support'] for m in metrics_data]
                )),
                'recall': float(np.average(
                    [m['recall'] for m in metrics_data],
                    weights=[m['support'] for m in metrics_data]
                )),
                'f1_score': float(np.average(
                    [m['f1_score'] for m in metrics_data],
                    weights=[m['support'] for m in metrics_data]
                ))
            }
        }
        
        with open(os.path.join(self.results_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed classification report
        with open(os.path.join(self.results_dir, 'classification_report.txt'), 'w') as f:
            f.write("EMOTION DETECTION SYSTEM - CLASSIFICATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Data Path: {self.test_data_path}\n")
            f.write(f"Total Samples: {len(true_labels)}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write("=" * 80 + "\n\n")
            f.write(classification_report)
            f.write("\n\n")
        
        # Save confusion matrix as CSV
        cm_df_data = []
        for i, true_emotion in enumerate(self.emotions):
            row = [true_emotion.capitalize()]
            row.extend(confusion_matrix[i].tolist())
            cm_df_data.append(row)
        
        with open(os.path.join(self.results_dir, 'confusion_matrix.csv'), 'w') as f:
            # Header
            f.write('True\\Predicted,' + ','.join([e.capitalize() for e in self.emotions]) + '\n')
            # Data
            for row in cm_df_data:
                f.write(','.join(map(str, row)) + '\n')
        
        # Save raw predictions for further analysis
        predictions_data = {
            'true_labels': true_labels.tolist(),
            'predicted_labels': predicted_labels.tolist(),
            'confidences': confidences,
            'failed_predictions': failed_predictions
        }
        
        with open(os.path.join(self.results_dir, 'raw_predictions.json'), 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        print(f"  ✓ Saved evaluation_summary.json")
        print(f"  ✓ Saved classification_report.txt")
        print(f"  ✓ Saved confusion_matrix.csv")
        print(f"  ✓ Saved raw_predictions.json")
    
    def generate_visualizations(self, cm, metrics_data):
        """Generate visualization plots"""
        print("\nGenerating visualizations...")
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[e.capitalize() for e in self.emotions],
            yticklabels=[e.capitalize() for e in self.emotions],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Emotion Detection System', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved confusion_matrix.png")
        
        # 2. Normalized Confusion Matrix (percentages)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='RdYlGn',
            xticklabels=[e.capitalize() for e in self.emotions],
            yticklabels=[e.capitalize() for e in self.emotions],
            vmin=0, 
            vmax=1,
            cbar_kws={'label': 'Percentage'}
        )
        plt.title('Normalized Confusion Matrix - Emotion Detection System', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved confusion_matrix_normalized.png")
        
        # 3. Precision, Recall, F1-Score Bar Chart
        emotions = [m['emotion'].capitalize() for m in metrics_data]
        precision = [m['precision'] for m in metrics_data]
        recall = [m['recall'] for m in metrics_data]
        f1 = [m['f1_score'] for m in metrics_data]
        
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
        
        ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics by Emotion', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved metrics_comparison.png")
        
        # 4. Support Distribution (number of samples per emotion)
        support = [m['support'] for m in metrics_data]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(emotions, support, color='#9b59b6')
        ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title('Test Dataset Distribution', fontsize=16, fontweight='bold')
        ax.set_xticklabels(emotions, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved dataset_distribution.png")
        
        # 5. Performance Summary Dashboard
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Confusion matrix (top left)
        ax1 = fig.add_subplot(gs[0:2, 0])
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='RdYlGn',
            xticklabels=[e.capitalize() for e in self.emotions],
            yticklabels=[e.capitalize() for e in self.emotions],
            vmin=0, 
            vmax=1,
            ax=ax1,
            cbar_kws={'label': 'Accuracy'}
        )
        ax1.set_title('Confusion Matrix (Normalized)', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # Metrics comparison (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        x_pos = np.arange(len(emotions))
        ax2.plot(x_pos, precision, 'o-', label='Precision', linewidth=2, markersize=8)
        ax2.plot(x_pos, recall, 's-', label='Recall', linewidth=2, markersize=8)
        ax2.plot(x_pos, f1, '^-', label='F1-Score', linewidth=2, markersize=8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(emotions, rotation=45, ha='right')
        ax2.set_ylabel('Score')
        ax2.set_title('Performance Metrics', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        # Dataset distribution (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.barh(emotions, support, color='#3498db', alpha=0.7)
        ax3.set_xlabel('Number of Samples')
        ax3.set_title('Dataset Distribution', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Overall statistics (bottom)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        overall_accuracy = np.trace(cm) / np.sum(cm)
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        stats_text = f"""
        OVERALL PERFORMANCE SUMMARY
        ══════════════════════════════════════════════════════════════════════════════
        
        Overall Accuracy:    {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)
        
        Macro Average:       Precision: {macro_precision:.4f}  |  Recall: {macro_recall:.4f}  |  F1-Score: {macro_f1:.4f}
        
        Total Samples:       {np.sum(support)}
        Evaluation Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax4.text(0.5, 0.5, stats_text, ha='center', va='center', 
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle('Emotion Detection System - Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(os.path.join(self.results_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved performance_dashboard.png")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Emotion Detection System')
    parser.add_argument('--test-path', type=str, default='data/combined_dataset_complete/validation',
                       help='Path to test data directory')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Total number of samples to evaluate (default: all)')
    parser.add_argument('--min-per-class', type=int, default=50,
                       help='Minimum samples per emotion class (default: 50)')
    parser.add_argument('--max-per-class', type=int, default=200,
                       help='Maximum samples per emotion class (default: 200)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: evaluate 100 samples per class (700 total)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: evaluate 50 samples per class (350 total)')
    
    args = parser.parse_args()
    
    # Set test data path
    test_data_path = args.test_path
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test data path not found: {test_data_path}")
        return
    
    # Determine sampling parameters
    if args.quick:
        print("\n🚀 QUICK MODE: Evaluating 50 samples per class (~350 total)\n")
        min_samples = 50
        max_samples = 50
    elif args.fast:
        print("\n⚡ FAST MODE: Evaluating 100 samples per class (~700 total)\n")
        min_samples = 100
        max_samples = 100
    else:
        min_samples = args.min_per_class
        max_samples = args.max_per_class
        if max_samples < 1000:
            print(f"\n📊 BALANCED MODE: Evaluating up to {max_samples} samples per class\n")
        else:
            print(f"\n📊 FULL EVALUATION: Processing all available data\n")
    
    # Create evaluator
    evaluator = EmotionSystemEvaluator(
        test_data_path,
        sample_size=args.sample_size,
        min_samples_per_class=min_samples,
        max_samples_per_class=max_samples
    )
    
    # Run evaluation
    try:
        accuracy, metrics, cm = evaluator.evaluate()
        print("\n" + "="*60)
        print("✓ EVALUATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nResults saved to: {evaluator.results_dir}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
