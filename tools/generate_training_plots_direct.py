#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
HISTORY_PATH = os.path.join(ROOT, 'logs', 'training_revised_20260220_030450', 'training_history.json')
OUTPUT_DIR = os.path.join(ROOT, 'evaluation_results', 'revised_cnn_eval_20260223_121202_tta10')
MODEL_TITLE = 'emotion_revised_cnn_20260220_030450.keras (TTAx10)'

with open(HISTORY_PATH, 'r') as f:
    payload = json.load(f)

history = payload.get('history', payload)
acc = history.get('accuracy')
val_acc = history.get('val_accuracy')
loss = history.get('loss')
val_loss = history.get('val_loss')

if not (acc and val_acc and loss and val_loss):
    raise SystemExit('Missing required history keys')

epochs = list(range(len(acc)))

old_rc = plt.rcParams.copy()
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 16,
})
try:
    # Accuracy
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, acc, label='train', linewidth=2)
    ax.plot(epochs, val_acc, label='valid', linewidth=2)
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(f'Training Accuracy - {MODEL_TITLE}')
    fig.tight_layout(rect=[0,0,1,0.95])
    out_acc = os.path.join(OUTPUT_DIR, 'training_accuracy.png')
    fig.savefig(out_acc, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Loss
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(epochs, loss, label='train', linewidth=2)
    ax.plot(epochs, val_loss, label='valid', linewidth=2)
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle(f'Training Loss - {MODEL_TITLE}')
    fig.tight_layout(rect=[0,0,1,0.95])
    out_loss = os.path.join(OUTPUT_DIR, 'training_loss.png')
    fig.savefig(out_loss, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('Saved:', out_acc, out_loss)
finally:
    plt.rcParams.update(old_rc)
