#!/usr/bin/env python3
"""
Quick evaluation script for emotion_simple_cnn model
Includes visualization similar to evaluate_system.py
"""

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support

def evaluate_simple_cnn():
    print('='*70)
    print('EVALUATING emotion_simple_cnn MODEL')
    print('='*70)
    
    # Load model
    model_path = 'models/emotion_simple_cnn_20251113_174858.h5'
    model = keras.models.load_model(model_path)
    print(f'✅ Model loaded: {os.path.basename(model_path)}')
    print(f'   Input: {model.input_shape}')
    print(f'   Output: {model.output_shape}')
    
    # Prepare validation data
    val_dir = 'data/combined_dataset_complete/validation'
    print(f'\n📁 Loading validation data from: {val_dir}')
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(96, 96),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f'✅ Found {val_generator.samples} validation images')
    print(f'   Classes: {list(val_generator.class_indices.keys())}')
    
    # Evaluate
    print(f'\n🔍 Evaluating model on validation set...')
    results = model.evaluate(val_generator, verbose=1)
    
    print(f'\n' + '='*70)
    print('OVERALL RESULTS')
    print('='*70)
    print(f'Loss: {results[0]:.4f}')
    print(f'Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)')
    
    # Get predictions for confusion matrix
    print(f'\n📊 Generating predictions for detailed analysis...')
    predictions = model.predict(val_generator, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    
    emotions = list(val_generator.class_indices.keys())
    
    # Per-class metrics
    print(f'\n' + '='*70)
    print('PER-CLASS ACCURACY')
    print('='*70)
    
    metrics_data = []
    for i, emotion in enumerate(emotions):
        mask = true_classes == i
        total = mask.sum()
        if total > 0:
            correct = (predicted_classes[mask] == i).sum()
            acc = correct / total
            print(f'{emotion.capitalize():10s}: {correct:4d}/{total:4d} = {acc:.4f} ({acc*100:.1f}%)')
            metrics_data.append({
                'emotion': emotion,
                'accuracy': float(acc),
                'correct': int(correct),
                'total': int(total)
            })
    
    # Calculate precision, recall, F1 for detailed metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        true_classes, 
        predicted_classes,
        labels=range(len(emotions)),
        zero_division=0
    )
    
    # Update metrics_data with precision, recall, f1
    for i, metric in enumerate(metrics_data):
        metric['precision'] = float(precision[i])
        metric['recall'] = float(recall[i])
        metric['f1_score'] = float(f1[i])
        metric['support'] = int(support[i])
    
    # Confusion matrix
    print(f'\n' + '='*70)
    print('CONFUSION MATRIX (rows=true, cols=predicted)')
    print('='*70)
    
    # Create confusion matrix
    cm = np.zeros((len(emotions), len(emotions)), dtype=int)
    for true_idx, pred_idx in zip(true_classes, predicted_classes):
        cm[true_idx][pred_idx] += 1
    
    # Print header
    header = '           '
    for emotion in emotions:
        header += f'{emotion[:4]:>6s} '
    print(header)
    print('-' * 70)
    
    # Print matrix
    for i, emotion in enumerate(emotions):
        row = f'{emotion.capitalize():10s} '
        for j in range(len(emotions)):
            row += f'{cm[i][j]:6d} '
        print(row)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'evaluation_results/simple_cnn_eval_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary = {
        'model': os.path.basename(model_path),
        'timestamp': timestamp,
        'validation_samples': int(val_generator.samples),
        'overall_loss': float(results[0]),
        'overall_accuracy': float(results[1]),
        'emotions': emotions,
        'per_class_metrics': metrics_data,
        'macro_avg': {
            'precision': float(np.mean(precision)),
            'recall': float(np.mean(recall)),
            'f1_score': float(np.mean(f1))
        },
        'weighted_avg': {
            'precision': float(np.average(precision, weights=support)),
            'recall': float(np.average(recall, weights=support)),
            'f1_score': float(np.average(f1, weights=support))
        }
    }
    
    with open(f'{output_dir}/evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save confusion matrix
    np.savetxt(f'{output_dir}/confusion_matrix.csv', cm, 
               delimiter=',', fmt='%d',
               header=','.join(emotions))
    
    # Save detailed predictions
    with open(f'{output_dir}/predictions.txt', 'w') as f:
        f.write('True,Predicted,Confidence\n')
        for true_idx, pred_idx, probs in zip(true_classes, predicted_classes, predictions):
            conf = probs[pred_idx]
            f.write(f'{emotions[true_idx]},{emotions[pred_idx]},{conf:.4f}\n')
    
    print(f'\n✅ Results saved to: {output_dir}/')
    
    # Generate visualizations
    print(f'\n📊 Generating visualizations...')
    generate_visualizations(cm, emotions, results, output_dir, metrics_data)
    
    print('='*70)
    
    return summary

def generate_visualizations(cm, emotions, results, output_dir, metrics_data):
    """Generate visualization plots similar to evaluate_system.py"""
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[e.capitalize() for e in emotions],
        yticklabels=[e.capitalize() for e in emotions],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - emotion_simple_cnn Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
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
        xticklabels=[e.capitalize() for e in emotions],
        yticklabels=[e.capitalize() for e in emotions],
        vmin=0, 
        vmax=1,
        cbar_kws={'label': 'Percentage'}
    )
    plt.title('Normalized Confusion Matrix - emotion_simple_cnn Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved confusion_matrix_normalized.png")
    
    # 3. Precision, Recall, F1-Score Bar Chart
    emotion_labels = [m['emotion'].capitalize() for m in metrics_data]
    precision = [m['precision'] for m in metrics_data]
    recall = [m['recall'] for m in metrics_data]
    f1 = [m['f1_score'] for m in metrics_data]
    
    x = np.arange(len(emotion_labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')
    
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics by Emotion - emotion_simple_cnn', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_labels, rotation=45, ha='right')
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
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved metrics_comparison.png")
    
    # 4. Support Distribution (number of samples per emotion)
    support = [m['support'] for m in metrics_data]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(emotion_labels, support, color='#9b59b6')
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Validation Dataset Distribution', fontsize=16, fontweight='bold')
    ax.set_xticklabels(emotion_labels, rotation=45, ha='right')
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
    plt.savefig(os.path.join(output_dir, 'dataset_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved dataset_distribution.png")
    
    # 5. Per-Class Accuracy Bar Chart (sorted)
    accuracies = [m['accuracy'] for m in metrics_data]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_emotions = [emotion_labels[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    # Color code: green (>70%), yellow (50-70%), red (<50%)
    colors = []
    for acc in sorted_accuracies:
        if acc >= 0.70:
            colors.append('#2ecc71')  # Green
        elif acc >= 0.50:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(sorted_emotions, sorted_accuracies, color=colors)
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy (Sorted) - emotion_simple_cnn', fontsize=16, fontweight='bold')
    ax.set_xticklabels(sorted_emotions, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='70% threshold')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_emotion.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved accuracy_by_emotion.png")
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")

if __name__ == '__main__':
    evaluate_simple_cnn()
