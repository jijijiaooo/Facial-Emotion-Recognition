#!/usr/bin/env python3
"""
Quick comparison script to run both TensorFlow and MindSpore training
and compare their performance.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    dependencies = {
        'tensorflow': False,
        'mindspore': False,
        'numpy': False,
        'pillow': False
    }
    
    try:
        import tensorflow
        dependencies['tensorflow'] = True
        print(f"  ✓ TensorFlow {tensorflow.__version__}")
    except ImportError:
        print("  ✗ TensorFlow not found")
    
    try:
        import mindspore
        dependencies['mindspore'] = True
        print(f"  ✓ MindSpore {mindspore.__version__}")
    except ImportError:
        print("  ✗ MindSpore not found")
    
    try:
        import numpy
        dependencies['numpy'] = True
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError:
        print("  ✗ NumPy not found")
    
    try:
        import PIL
        dependencies['pillow'] = True
        print(f"  ✓ Pillow {PIL.__version__}")
    except ImportError:
        print("  ✗ Pillow not found")
    
    return dependencies

def install_mindspore():
    """Install MindSpore"""
    print("\nInstalling MindSpore...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mindspore"])
        print("  ✓ MindSpore installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("  ✗ Failed to install MindSpore")
        return False

def run_training(script_name, epochs=10, batch_size=64):
    """Run a training script"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        script_name,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--train_dir', 'data/Emotion_Classification/train',
        '--val_dir', 'data/Emotion_Classification/validation'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with error code {e.returncode}")
        return False

def compare_results():
    """Compare results from both training runs"""
    print(f"\n{'='*80}")
    print("COMPARING RESULTS")
    print(f"{'='*80}")
    
    # Find latest log directories
    logs_dir = Path('logs')
    
    # Find TensorFlow logs
    tf_logs = sorted(logs_dir.glob('training_2*'), key=os.path.getmtime, reverse=True)
    tf_latest = tf_logs[0] if tf_logs else None
    
    # Find MindSpore logs
    ms_logs = sorted(logs_dir.glob('training_mindspore_*'), key=os.path.getmtime, reverse=True)
    ms_latest = ms_logs[0] if ms_logs else None
    
    results = {}
    
    if tf_latest:
        tf_history_file = tf_latest / 'training_history.json'
        if tf_history_file.exists():
            with open(tf_history_file) as f:
                tf_history = json.load(f)
                results['tensorflow'] = {
                    'final_val_accuracy': tf_history['val_accuracy'][-1] if tf_history.get('val_accuracy') else 0,
                    'final_val_loss': tf_history['val_loss'][-1] if tf_history.get('val_loss') else 0,
                    'log_dir': str(tf_latest)
                }
    
    if ms_latest:
        ms_results_file = ms_latest / 'final_results.json'
        if ms_results_file.exists():
            with open(ms_results_file) as f:
                ms_results = json.load(f)
                results['mindspore'] = {
                    'final_val_accuracy': ms_results.get('final_val_accuracy', 0),
                    'log_dir': str(ms_latest)
                }
    
    # Display comparison
    print("\n" + "-"*80)
    print(f"{'Framework':<20} {'Val Accuracy':<15} {'Log Directory'}")
    print("-"*80)
    
    if 'tensorflow' in results:
        tf_res = results['tensorflow']
        print(f"{'TensorFlow':<20} {tf_res['final_val_accuracy']*100:>6.2f}%        {tf_res['log_dir']}")
    else:
        print(f"{'TensorFlow':<20} {'N/A':<15} {'Not found'}")
    
    if 'mindspore' in results:
        ms_res = results['mindspore']
        print(f"{'MindSpore':<20} {ms_res['final_val_accuracy']*100:>6.2f}%        {ms_res['log_dir']}")
    else:
        print(f"{'MindSpore':<20} {'N/A':<15} {'Not found'}")
    
    print("-"*80)
    
    if 'tensorflow' in results and 'mindspore' in results:
        tf_acc = results['tensorflow']['final_val_accuracy']
        ms_acc = results['mindspore']['final_val_accuracy']
        diff = (ms_acc - tf_acc) * 100
        
        print(f"\nImprovement: {diff:+.2f}%")
        
        if diff > 0:
            print(f"✓ MindSpore model is {diff:.2f}% better")
        elif diff < 0:
            print(f"✗ MindSpore model is {abs(diff):.2f}% worse")
        else:
            print("= Both models have the same performance")

def main():
    print("="*80)
    print("EMOTION RECOGNITION TRAINING COMPARISON")
    print("TensorFlow vs MindSpore")
    print("="*80)
    
    # Check dependencies
    deps = check_dependencies()
    
    # Install MindSpore if needed
    if not deps['mindspore']:
        print("\nMindSpore is not installed.")
        response = input("Would you like to install it now? (y/n): ").strip().lower()
        if response == 'y':
            if not install_mindspore():
                print("\nCannot proceed without MindSpore. Exiting.")
                return
        else:
            print("\nCannot run MindSpore training without MindSpore. Exiting.")
            return
    
    # Check if TensorFlow is available
    if not deps['tensorflow']:
        print("\nWARNING: TensorFlow is not installed.")
        print("Only MindSpore training will be run.")
    
    # Get training parameters
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    
    try:
        epochs = int(input("Number of epochs (default 10): ") or "10")
        batch_size = int(input("Batch size (default 64): ") or "64")
    except ValueError:
        print("Invalid input. Using defaults: epochs=10, batch_size=64")
        epochs = 10
        batch_size = 64
    
    print(f"\nConfiguration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    response = input("\nProceed with training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Run trainings
    results = {}
    
    if deps['tensorflow']:
        results['tensorflow'] = run_training(
            'src/core/train_simple_cnn.py',
            epochs=epochs,
            batch_size=batch_size
        )
    
    if deps['mindspore']:
        results['mindspore'] = run_training(
            'src/core/train_mindspore_cnn.py',
            epochs=epochs,
            batch_size=batch_size
        )
    
    # Compare results
    if any(results.values()):
        compare_results()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
