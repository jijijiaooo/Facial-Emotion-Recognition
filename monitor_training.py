#!/usr/bin/env python3
"""Monitor training progress from JSON logs"""

import os
import json
import glob
from datetime import datetime

def get_latest_log():
    log_dirs = glob.glob('logs/training_*')
    if not log_dirs:
        return None
    latest = max(log_dirs, key=os.path.getmtime)
    return os.path.join(latest, 'training_history.json')

def monitor():
    log_file = get_latest_log()
    if not log_file:
        print("No training logs found")
        return
    
    if not os.path.exists(log_file):
        print(f"Log file not created yet: {log_file}")
        print("Training is still in progress...")
        return
    
    with open(log_file) as f:
        history = json.load(f)
    
    print("=" * 70)
    print("TRAINING PROGRESS SUMMARY")
    print("=" * 70)
    
    for phase in ['phase1', 'phase2']:
        if phase not in history:
            continue
        
        phase_data = history[phase]
        if not phase_data or 'accuracy' not in phase_data:
            continue
        
        epochs = len(phase_data['accuracy'])
        if epochs == 0:
            continue
        
        print(f"\n{phase.upper()}:")
        print(f"  Epochs completed: {epochs}")
        print(f"  Training accuracy: {phase_data['accuracy'][-1]:.4f} (best: {max(phase_data['accuracy']):.4f})")
        print(f"  Validation accuracy: {phase_data['val_accuracy'][-1]:.4f} (best: {max(phase_data['val_accuracy']):.4f})")
        print(f"  Training loss: {phase_data['loss'][-1]:.4f} (best: {min(phase_data['loss']):.4f})")
        print(f"  Validation loss: {phase_data['val_loss'][-1]:.4f} (best: {min(phase_data['val_loss']):.4f})")
        
        if 'learning_rate' in phase_data and phase_data['learning_rate']:
            print(f"  Current LR: {phase_data['learning_rate'][-1]:.2e}")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    monitor()
