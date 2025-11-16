#!/usr/bin/env python3
"""
Evaluate MindSpore Emotion Recognition Model
Loads a trained MindSpore checkpoint and evaluates on test/validation data
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np

import mindspore
from mindspore import nn, context, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.dataset import ImageFolderDataset
import mindspore.dataset.vision as vision_transforms
import mindspore.dataset.transforms as data_transforms
from mindspore.common import dtype as mstype

# Import model architecture
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.core.train_mindspore_cnn import EnhancedEmotionCNN

# Set context
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def create_eval_dataset(data_dir, img_size=96, batch_size=64, num_parallel_workers=4):
    """Create dataset for evaluation"""
    
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    dataset = ImageFolderDataset(
        data_dir,
        class_indexing={label: idx for idx, label in enumerate(emotion_labels)},
        num_parallel_workers=num_parallel_workers,
        shuffle=False
    )
    
    # Preprocessing pipeline (same as validation in training)
    transform_list = [
        vision_transforms.Decode(),
        vision_transforms.Grayscale(1),
        vision_transforms.Resize((img_size, img_size)),
        vision_transforms.Rescale(1.0 / 255.0, 0.0),
        vision_transforms.Normalize([0.5], [0.5]),
        vision_transforms.HWC2CHW()
    ]
    
    dataset = dataset.map(
        operations=transform_list,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers
    )
    
    # Type cast labels
    type_cast_op = data_transforms.TypeCast(mstype.int32)
    dataset = dataset.map(
        operations=type_cast_op,
        input_columns=["label"],
        num_parallel_workers=num_parallel_workers
    )
    
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    return dataset


def compute_per_class_accuracy(network, dataset, num_classes=7):
    """Compute per-class accuracy and confusion matrix"""
    
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # Iterate through dataset
    for batch in dataset.create_dict_iterator():
        images = batch['image']
        labels = batch['label'].asnumpy()
        
        # Forward pass
        predictions = network(images)
        pred_labels = np.argmax(predictions.asnumpy(), axis=1)
        
        # Update confusion matrix
        for true_label, pred_label in zip(labels, pred_labels):
            confusion_matrix[true_label][pred_label] += 1
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for i, label in enumerate(emotion_labels):
        total = confusion_matrix[i].sum()
        correct = confusion_matrix[i][i]
        accuracy = correct / total if total > 0 else 0.0
        per_class_acc[label] = {
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(total)
        }
    
    return per_class_acc, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MindSpore Emotion Model')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--data_dir', required=True,
                       help='Directory containing test/validation data')
    parser.add_argument('--img_size', type=int, default=96,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_classes', type=int, default=7,
                       help='Number of emotion classes')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (should match training)')
    parser.add_argument('--output_dir', default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'mindspore_eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    print('=' * 80)
    print('MINDSPORE MODEL EVALUATION')
    print('=' * 80)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Data directory: {args.data_dir}')
    print(f'Image size: {args.img_size}')
    print(f'Batch size: {args.batch_size}')
    print('=' * 80)
    
    # Step 1: Create dataset
    print('\n[1/4] Creating evaluation dataset...')
    eval_dataset = create_eval_dataset(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_parallel_workers=args.num_workers
    )
    
    dataset_size = eval_dataset.get_dataset_size()
    print(f'✓ Dataset size: {dataset_size} batches')
    
    # Step 2: Build model
    print('\n[2/4] Building model...')
    network = EnhancedEmotionCNN(num_classes=args.num_classes, dropout_rate=args.dropout)
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    param_dict = load_checkpoint(args.checkpoint)
    load_param_into_net(network, param_dict)
    network.set_train(False)  # Set to evaluation mode
    
    print('✓ Model loaded successfully')
    
    # Step 3: Evaluate
    print('\n[3/4] Evaluating model...')
    
    # Overall accuracy
    metrics = {'accuracy': Accuracy()}
    model = Model(network, metrics=metrics)
    
    result = model.eval(eval_dataset, dataset_sink_mode=False)
    overall_accuracy = result['accuracy']
    
    print(f'\n✓ Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)')
    
    # Step 4: Compute detailed metrics
    print('\n[4/4] Computing per-class metrics...')
    per_class_acc, confusion_matrix = compute_per_class_accuracy(
        network, eval_dataset, num_classes=args.num_classes
    )
    
    print('\nPer-Class Accuracy:')
    print('-' * 60)
    print(f'{"Emotion":<12} {"Accuracy":<12} {"Correct":<10} {"Total":<10}')
    print('-' * 60)
    
    for label in emotion_labels:
        if label in per_class_acc:
            acc_info = per_class_acc[label]
            print(f'{label:<12} {acc_info["accuracy"]*100:>6.2f}%      '
                  f'{acc_info["correct"]:<10} {acc_info["total"]:<10}')
    
    print('-' * 60)
    
    # Save results
    print('\nSaving results...')
    
    # Save confusion matrix
    confusion_file = os.path.join(output_dir, 'confusion_matrix.csv')
    np.savetxt(confusion_file, confusion_matrix, delimiter=',', fmt='%d',
               header=','.join(emotion_labels), comments='')
    print(f'✓ Confusion matrix: {confusion_file}')
    
    # Save summary
    summary = {
        'overall_accuracy': float(overall_accuracy),
        'per_class_accuracy': per_class_acc,
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'num_classes': args.num_classes,
        'timestamp': timestamp
    }
    
    summary_file = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'✓ Evaluation summary: {summary_file}')
    
    # Save detailed predictions (optional)
    print('\nGenerating detailed predictions...')
    predictions_file = os.path.join(output_dir, 'predictions.txt')
    
    with open(predictions_file, 'w') as f:
        f.write('Detailed Predictions\n')
        f.write('=' * 80 + '\n\n')
        
        batch_num = 0
        for batch in eval_dataset.create_dict_iterator():
            batch_num += 1
            images = batch['image']
            labels = batch['label'].asnumpy()
            
            predictions = network(images)
            pred_labels = np.argmax(predictions.asnumpy(), axis=1)
            confidences = np.max(predictions.asnumpy(), axis=1)
            
            for i, (true_label, pred_label, conf) in enumerate(zip(labels, pred_labels, confidences)):
                status = '✓' if true_label == pred_label else '✗'
                f.write(f'{status} True: {emotion_labels[true_label]:<10} '
                       f'Pred: {emotion_labels[pred_label]:<10} '
                       f'Confidence: {conf:.4f}\n')
            
            if batch_num % 10 == 0:
                print(f'  Processed {batch_num}/{dataset_size} batches...', end='\r')
        
        print(f'  Processed {batch_num}/{dataset_size} batches... Done!')
    
    print(f'✓ Detailed predictions: {predictions_file}')
    
    print('\n' + '=' * 80)
    print('EVALUATION COMPLETED!')
    print('=' * 80)
    print(f'Results saved to: {output_dir}')
    print(f'Overall Accuracy: {overall_accuracy*100:.2f}%')
    print('=' * 80)


if __name__ == '__main__':
    main()
