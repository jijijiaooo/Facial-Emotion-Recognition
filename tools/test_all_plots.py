#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(__file__))
HISTORY_PATH = os.path.join(ROOT, 'logs', 'training_revised_20260220_030450', 'training_history.json')
OUTPUT_DIR = os.path.join(ROOT, 'evaluation_results', 'revised_cnn_eval_20260223_121202_tta10_test')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_TITLE = 'emotion_revised_cnn_20260220_030450.keras (TTAx10)'
EMOTIONS = ["neutral", "happy", "sad", "angry", "disgust", "shocked"]

# Load training history
with open(HISTORY_PATH, 'r') as f:
    payload = json.load(f)
history = payload.get('history', payload)

# Create sample confusion matrix for demo
cm_sample = np.array([
    [28, 0, 2, 0, 0, 0],
    [1, 28, 0, 0, 1, 0],
    [3, 0, 26, 1, 0, 0],
    [1, 0, 2, 25, 2, 0],
    [1, 0, 0, 4, 25, 0],
    [0, 2, 0, 0, 0, 28],
], dtype=int)

# Test font application with rcParams
old_rc = plt.rcParams.copy()
try:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 16,
    })

    # Test confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_sample,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        annot_kws={'size': 18},
    )
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")

    # Test normalized confusion matrix
    cm_normalized = cm_sample.astype('float') / cm_sample.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        vmin=0,
        vmax=1,
        annot_kws={'size': 18},
    )
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    cm_norm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Normalized confusion matrix saved: {cm_norm_path}")

    # Test accuracy plot
    plt.figure(figsize=(12, 6))
    accuracies = [0.93, 0.96, 0.87, 0.88, 0.83, 0.93]
    emotion_names = [e.capitalize() for e in EMOTIONS]
    colors = ['#e74c3c' if acc < 0.7 else '#f39c12' if acc < 0.8 else '#27ae60' for acc in accuracies]

    bars = plt.bar(emotion_names, accuracies, color=colors, alpha=0.7)
    plt.axhline(y=0.9, color='blue', linestyle='--', label=f'Overall Accuracy: 90.0%')
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title(f'Per-Class Accuracy - {MODEL_TITLE}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    acc_path = os.path.join(OUTPUT_DIR, 'per_class_accuracy.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Per-class accuracy saved: {acc_path}")

    # Test precision/recall/f1 plot
    plt.figure(figsize=(14, 6))
    x = np.arange(len(EMOTIONS))
    width = 0.25
    precision = [0.92, 0.95, 0.86, 0.87, 0.82, 0.92]
    recall = [0.93, 0.97, 0.88, 0.89, 0.84, 0.94]
    f1 = [0.92, 0.96, 0.87, 0.88, 0.83, 0.93]
    
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    plt.ylabel('Score')
    plt.title(f'Precision, Recall, and F1-Score by Emotion - {MODEL_TITLE}')
    plt.xticks(x, emotion_names, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    metrics_path = os.path.join(OUTPUT_DIR, 'precision_recall_f1.png')
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision/Recall/F1 saved: {metrics_path}")

    # Test training curves
    acc = history.get('accuracy')
    val_acc = history.get('val_accuracy')
    loss = history.get('loss')
    val_loss = history.get('val_loss')
    
    if acc and val_acc and loss and val_loss:
        epochs = list(range(len(acc)))
        
        # Accuracy plot
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(epochs, acc, label='train', linewidth=2)
        ax.plot(epochs, val_acc, label='valid', linewidth=2)
        ax.set_title('Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.suptitle(f'Training Accuracy - {MODEL_TITLE}')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_acc = os.path.join(OUTPUT_DIR, 'training_accuracy.png')
        fig.savefig(out_acc, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Training accuracy saved: {out_acc}")

        # Loss plot
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(epochs, loss, label='train', linewidth=2)
        ax.plot(epochs, val_loss, label='valid', linewidth=2)
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.suptitle(f'Training Loss - {MODEL_TITLE}')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_loss = os.path.join(OUTPUT_DIR, 'training_loss.png')
        fig.savefig(out_loss, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Training loss saved: {out_loss}")

finally:
    plt.rcParams.update(old_rc)

print(f"\n✅ All test plots generated in {OUTPUT_DIR}")
