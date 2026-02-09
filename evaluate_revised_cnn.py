#!/usr/bin/env python3
"""
Evaluation script for emotion_revised_cnn model (6 classes)
Evaluates on the revised dataset without 'surprise' emotion
"""

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

def evaluate_revised_cnn():
    print('='*70)
    print('EVALUATING REVISED CNN MODEL (6 CLASSES)')
    print('='*70)
    
    # Load model
    model_path = 'models/emotion_revised_cnn_20260205_172616.keras'
    print(f'Loading model: {model_path}')
    model = keras.models.load_model(model_path)
    print(f'✅ Model loaded: {os.path.basename(model_path)}')
    print(f'   Input: {model.input_shape}')
    print(f'   Output: {model.output_shape}')
    
    # Prepare validation data
    val_dir = 'data/revised_dataset/validation'
    print(f'\n📁 Loading validation data from: {val_dir}')
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(112, 112),
        color_mode='grayscale',
        batch_size=128,
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
    
    # Print detailed metrics
    print(f'\n' + '='*70)
    print('DETAILED METRICS (Precision, Recall, F1-Score)')
    print('='*70)
    print(f'{"Emotion":<10} {"Precision":<10} {"Recall":<10} {"F1-Score":<10} {"Support":<10}')
    print('-'*70)
    for metric in metrics_data:
        print(f'{metric["emotion"].capitalize():<10} {metric["precision"]:<10.4f} {metric["recall"]:<10.4f} {metric["f1_score"]:<10.4f} {metric["support"]:<10}')
    
    # Confusion matrix
    print(f'\n📊 Generating confusion matrix...')
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'evaluation_results/revised_cnn_eval_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix (raw counts)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix - Revised CNN Model (6 Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f'✅ Confusion matrix saved: {cm_path}')
    plt.close()
    
    # Plot normalized confusion matrix (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=emotions, yticklabels=emotions, vmin=0, vmax=1)
    plt.title('Normalized Confusion Matrix - Revised CNN Model (6 Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    print(f'✅ Normalized confusion matrix saved: {cm_norm_path}')
    plt.close()
    
    # Plot per-class accuracy bar chart
    plt.figure(figsize=(12, 6))
    accuracies = [m['accuracy'] for m in metrics_data]
    emotion_names = [m['emotion'].capitalize() for m in metrics_data]
    colors = ['#e74c3c' if acc < 0.7 else '#f39c12' if acc < 0.8 else '#27ae60' for acc in accuracies]
    
    bars = plt.bar(emotion_names, accuracies, color=colors, alpha=0.7)
    plt.axhline(y=results[1], color='blue', linestyle='--', label=f'Overall Accuracy: {results[1]:.2%}')
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy - Revised CNN Model (6 Classes)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    acc_path = os.path.join(output_dir, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    print(f'✅ Per-class accuracy chart saved: {acc_path}')
    plt.close()
    
    # Plot precision, recall, F1 comparison
    plt.figure(figsize=(14, 6))
    x = np.arange(len(emotions))
    width = 0.25
    
    plt.bar(x - width, [m['precision'] for m in metrics_data], width, label='Precision', alpha=0.8)
    plt.bar(x, [m['recall'] for m in metrics_data], width, label='Recall', alpha=0.8)
    plt.bar(x + width, [m['f1_score'] for m in metrics_data], width, label='F1-Score', alpha=0.8)
    
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score by Emotion - Revised CNN Model')
    plt.xticks(x, emotion_names, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    metrics_path = os.path.join(output_dir, 'precision_recall_f1.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f'✅ Metrics comparison chart saved: {metrics_path}')
    plt.close()
    
    # Save results to JSON
    results_dict = {
        'model_path': model_path,
        'model_name': os.path.basename(model_path),
        'evaluation_date': timestamp,
        'dataset': 'revised_dataset (6 classes)',
        'validation_samples': int(val_generator.samples),
        'overall_loss': float(results[0]),
        'overall_accuracy': float(results[1]),
        'emotions': emotions,
        'per_class_metrics': metrics_data,
        'confusion_matrix': cm.tolist()
    }
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f'✅ Results saved: {json_path}')
    
    # Generate classification report
    report = classification_report(
        true_classes, 
        predicted_classes,
        target_names=emotions,
        digits=4
    )
    
    print(f'\n' + '='*70)
    print('CLASSIFICATION REPORT')
    print('='*70)
    print(report)
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write('REVISED CNN MODEL EVALUATION REPORT\n')
        f.write('='*70 + '\n\n')
        f.write(f'Model: {model_path}\n')
        f.write(f'Evaluation Date: {timestamp}\n')
        f.write(f'Dataset: revised_dataset (6 classes)\n')
        f.write(f'Validation Samples: {val_generator.samples}\n\n')
        f.write('OVERALL RESULTS\n')
        f.write('-'*70 + '\n')
        f.write(f'Loss: {results[0]:.4f}\n')
        f.write(f'Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)\n\n')
        f.write('CLASSIFICATION REPORT\n')
        f.write('-'*70 + '\n')
        f.write(report)
    
    print(f'✅ Classification report saved: {report_path}')
    
    print(f'\n' + '='*70)
    print('EVALUATION COMPLETE')
    print('='*70)
    print(f'📁 All results saved to: {output_dir}')
    print(f'\nFiles generated:')
    print(f'  - confusion_matrix.png')
    print(f'  - confusion_matrix_normalized.png')
    print(f'  - per_class_accuracy.png')
    print(f'  - precision_recall_f1.png')
    print(f'  - results.json')
    print(f'  - classification_report.txt')
    
    return results_dict

if __name__ == '__main__':
    evaluate_revised_cnn()
