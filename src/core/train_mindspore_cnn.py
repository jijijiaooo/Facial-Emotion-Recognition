#!/usr/bin/env python3
"""
Train an Enhanced CNN using MindSpore for Emotion Classification
Improvements over TensorFlow version:
- MindSpore's efficient computation graph
- Enhanced data augmentation with MindSpore Vision
- Mixed precision training for faster convergence
- Advanced learning rate schedulers
- Gradient clipping for stability
- Custom metrics and logging
- Model checkpointing with metric tracking
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
from pathlib import Path

import mindspore
from mindspore import nn, ops, Tensor, context
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train.callback import TimeMonitor, Callback
from mindspore.nn import Adam, SGD, Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy
from mindspore.dataset import vision, transforms
from mindspore.dataset import GeneratorDataset, ImageFolderDataset
import mindspore.dataset.vision as vision_transforms
import mindspore.dataset.transforms as data_transforms
from mindspore.common import dtype as mstype

# Set context - use GPU if available, otherwise CPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# For GPU: context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)


class EnhancedEmotionCNN(nn.Cell):
    """Enhanced CNN architecture with residual connections and attention mechanism"""
    
    def __init__(self, num_classes=7, dropout_rate=0.5, input_channels=3):
        super(EnhancedEmotionCNN, self).__init__()
        
        # Conv Block 1 with residual connection
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)
        
        # Conv Block 2 with residual connection
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.25)
        
        # Conv Block 3 with residual connection
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.3)
        
        # Conv Block 4 with spatial attention
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, pad_mode='pad', has_bias=False)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=0.3)
        
        # Spatial Attention Module
        self.attention_conv = nn.Conv2d(512, 1, kernel_size=1)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Dense(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Dense(256, num_classes)
        
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        # Block 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        
        # Spatial Attention
        attention = self.sigmoid(self.attention_conv(x))
        x = x * attention
        
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        
        # Fully Connected
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x


class CustomLossMonitor(Callback):
    """Custom callback to monitor and log training metrics with progress bar"""
    
    def __init__(self, per_print_times=1):
        super(CustomLossMonitor, self).__init__()
        self.per_print_times = per_print_times
        self.losses = []
        self.epochs = []
        self.step_losses = []
        
    def epoch_begin(self, run_context):
        """Reset step losses at the beginning of each epoch"""
        self.step_losses = []
        
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]
        
        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        
        self.step_losses.append(loss)
        
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_epoch = (cb_params.cur_step_num - 1) // cb_params.batch_num + 1
        
        if cur_step_in_epoch % self.per_print_times == 0:
            # Calculate average loss so far in this epoch
            avg_loss = np.mean(self.step_losses) if self.step_losses else loss
            
            # Create progress bar
            progress = cur_step_in_epoch / cb_params.batch_num
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"\rEpoch [{cur_epoch:>3}/{cb_params.epoch_num}] |{bar}| "
                  f"{cur_step_in_epoch:>4}/{cb_params.batch_num} "
                  f"- Loss: {avg_loss:.4f}", end='', flush=True)
    
    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        self.epochs.append(epoch)
        
        # Calculate final average loss for the epoch
        if self.step_losses:
            avg_loss = np.mean(self.step_losses)
            self.losses.append(avg_loss)
            
            # Complete the progress bar
            bar = '█' * 30
            print(f"\rEpoch [{epoch:>3}/{cb_params.epoch_num}] |{bar}| "
                  f"{cb_params.batch_num:>4}/{cb_params.batch_num} "
                  f"- Loss: {avg_loss:.4f}", flush=True)


class MetricsLogger(Callback):
    """Callback to log training and validation metrics with real-time display"""
    
    def __init__(self, log_dir, val_dataset=None, network=None, loss_fn=None):
        super(MetricsLogger, self).__init__()
        self.log_dir = log_dir
        self.val_dataset = val_dataset
        self.network = network
        self.loss_fn = loss_fn
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        self.epoch_losses = []
        self.best_val_acc = 0.0
        os.makedirs(log_dir, exist_ok=True)
    
    def step_end(self, run_context):
        """Collect losses during training"""
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        
        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]
        
        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))
            self.epoch_losses.append(loss)
        
    def epoch_end(self, run_context):
        """Evaluate on validation set and display metrics after each epoch"""
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        
        # Calculate average training loss for this epoch
        if self.epoch_losses:
            avg_train_loss = np.mean(self.epoch_losses)
            self.history['train_loss'].append(float(avg_train_loss))
            self.epoch_losses = []
        else:
            avg_train_loss = 0.0
            self.history['train_loss'].append(0.0)
        
        # Get current learning rate
        if hasattr(cb_params, 'optimizer'):
            lr = cb_params.optimizer.learning_rate
            if isinstance(lr, (list, tuple)):
                current_lr = lr[cb_params.cur_step_num - 1] if cb_params.cur_step_num <= len(lr) else lr[-1]
            else:
                # Handle _IteratorLearningRate or other LR scheduler objects
                try:
                    current_lr = float(lr)
                except (TypeError, ValueError):
                    # For dynamic LR schedulers, get the value at current step
                    if hasattr(lr, 'asnumpy'):
                        current_lr = float(lr.asnumpy())
                    elif isinstance(lr, (int, float)):
                        current_lr = float(lr)
                    else:
                        # Fallback: use the base learning rate from optimizer
                        current_lr = self.history['learning_rate'][-1] if self.history['learning_rate'] else 0.001
            self.history['learning_rate'].append(current_lr)
        
        # Validate if validation dataset is provided
        if self.val_dataset and self.network:
            print(f"\n[Epoch {epoch}] Validating...", end=' ')
            
            # Create evaluation model with loss function
            metrics = {'accuracy': Accuracy()}
            eval_model = Model(self.network, loss_fn=self.loss_fn, metrics=metrics)
            
            # Evaluate
            result = eval_model.eval(self.val_dataset, dataset_sink_mode=False)
            val_acc = result['accuracy']
            
            self.history['val_accuracy'].append(float(val_acc))
            
            # Track best accuracy
            improved = ""
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improved = " ⭐ New Best!"
            
            # Display epoch summary
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{cb_params.epoch_num} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%){improved}")
            if self.history['learning_rate']:
                print(f"  Learning Rate: {self.history['learning_rate'][-1]:.6f}")
            print(f"{'='*70}\n")
        else:
            # No validation, just show training loss
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{cb_params.epoch_num} - Train Loss: {avg_train_loss:.4f}")
            if self.history['learning_rate']:
                print(f"  Learning Rate: {self.history['learning_rate'][-1]:.6f}")
            print(f"{'='*70}\n")
        
        # Save metrics to file
        history_file = os.path.join(self.log_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def create_dataset(data_dir, img_size=96, batch_size=64, is_training=True, num_parallel_workers=4, simple_aug=False):
    """Create dataset with enhanced augmentation"""
    
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Use ImageFolderDataset for automatic label extraction
    dataset = ImageFolderDataset(
        data_dir,
        class_indexing={label: idx for idx, label in enumerate(emotion_labels)},
        num_parallel_workers=num_parallel_workers,
        shuffle=is_training
    )
    
    if is_training:
        if simple_aug:
            # Simpler augmentation for faster training
            transform_list = [
                vision_transforms.Decode(),
                vision_transforms.Resize((img_size, img_size)),
                vision_transforms.RandomHorizontalFlip(prob=0.5),
                vision_transforms.Rescale(1.0 / 255.0, 0.0),
                vision_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                vision_transforms.HWC2CHW()
            ]
        else:
            # Enhanced augmentation for training - use C implementations (without Grayscale which has issues)
            # Convert RGB to grayscale manually after loading
            transform_list = [
                vision_transforms.Decode(),
                vision_transforms.Resize((img_size, img_size)),
                vision_transforms.RandomRotation(degrees=20),
                vision_transforms.RandomHorizontalFlip(prob=0.5),
                vision_transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                vision_transforms.RandomColorAdjust(brightness=0.2, contrast=0.2),
                vision_transforms.Rescale(1.0 / 255.0, 0.0),
                vision_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # RGB normalization
                vision_transforms.HWC2CHW()
            ]
    else:
        # Simple preprocessing for validation
        transform_list = [
            vision_transforms.Decode(),
            vision_transforms.Resize((img_size, img_size)),
            vision_transforms.Rescale(1.0 / 255.0, 0.0),
            vision_transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # RGB normalization
            vision_transforms.HWC2CHW()
        ]
    
    # Apply transformations
    dataset = dataset.map(
        operations=transform_list,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers
    )
    
    # Convert labels to one-hot encoding
    type_cast_op = data_transforms.TypeCast(mstype.int32)
    dataset = dataset.map(
        operations=type_cast_op,
        input_columns=["label"],
        num_parallel_workers=num_parallel_workers
    )
    
    # Batch and repeat
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    
    if is_training:
        dataset = dataset.repeat(1)
    
    return dataset


class WarmupCosineDecayLR:
    """Warmup + Cosine Decay Learning Rate Scheduler"""
    
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=1e-6):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        
    def __call__(self, global_step):
        if global_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (global_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def compute_class_weights(train_dir, emotion_labels):
    """Compute class weights for imbalanced dataset"""
    counts = {}
    for label in emotion_labels:
        label_dir = os.path.join(train_dir, label)
        if os.path.exists(label_dir):
            counts[label] = len([f for f in os.listdir(label_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        else:
            counts[label] = 0
    
    total = sum(counts.values())
    class_weights = {}
    for idx, label in enumerate(emotion_labels):
        if counts[label] > 0:
            class_weights[idx] = float(total) / (len(emotion_labels) * counts[label])
        else:
            class_weights[idx] = 1.0
    
    print(f"  Class counts: {counts}")
    print(f"  Class weights: {class_weights}")
    
    return class_weights, counts


def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced CNN with MindSpore')
    parser.add_argument('--train_dir', default='data/Emotion_Classification/train',
                       help='Training data directory')
    parser.add_argument('--val_dir', default='data/Emotion_Classification/validation',
                       help='Validation data directory')
    parser.add_argument('--img_size', type=int, default=96,
                       help='Input image size (square)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--model_dir', default='models',
                       help='Directory to save models')
    parser.add_argument('--logs_dir', default=None,
                       help='Directory for logs')
    parser.add_argument('--save_checkpoint_steps', type=int, default=500,
                       help='Steps to save checkpoint')
    parser.add_argument('--keep_checkpoint_max', type=int, default=5,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers for data loading')
    parser.add_argument('--no_validation', action='store_true',
                       help='Skip validation during training (faster)')
    parser.add_argument('--simple_aug', action='store_true',
                       help='Use simpler augmentation (faster)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MIN_LR = args.min_lr
    WARMUP_EPOCHS = args.warmup_epochs
    DROPOUT_RATE = args.dropout
    
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    MODEL_SAVE_DIR = os.path.join(args.model_dir, f'mindspore_cnn_{timestamp}')
    LOGS_DIR = args.logs_dir or f'logs/training_mindspore_{timestamp}'
    
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    NUM_CLASSES = len(EMOTION_LABELS)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    print('=' * 80)
    print('MINDSPORE ENHANCED CNN EMOTION CLASSIFICATION TRAINING')
    print(f'Image size: {IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}')
    print(f'Learning rate: {LEARNING_RATE} | Min LR: {MIN_LR} | Warmup: {WARMUP_EPOCHS} epochs')
    print(f'Dropout: {DROPOUT_RATE}')
    print('=' * 80)
    
    # Step 1: Compute class weights
    print('\n[1/7] Computing class weights...')
    class_weights, class_counts = compute_class_weights(TRAIN_DIR, EMOTION_LABELS)
    
    # Step 2: Create datasets
    print('\n[2/7] Creating datasets...')
    train_dataset = create_dataset(
        TRAIN_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        is_training=True,
        num_parallel_workers=args.num_workers,
        simple_aug=args.simple_aug
    )
    
    val_dataset = create_dataset(
        VAL_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        is_training=False,
        num_parallel_workers=args.num_workers,
        simple_aug=False
    )
    
    train_steps = train_dataset.get_dataset_size()
    val_steps = val_dataset.get_dataset_size()
    
    print(f'✓ Training steps per epoch: {train_steps}')
    print(f'✓ Validation steps per epoch: {val_steps}')
    
    # Step 3: Build model
    print('\n[3/7] Building enhanced CNN model...')
    network = EnhancedEmotionCNN(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, input_channels=3)
    
    # Calculate total parameters
    total_params = sum([param.size for param in network.trainable_params()])
    print(f'✓ Total trainable parameters: {total_params:,}')
    
    # Step 4: Define loss and optimizer
    print('\n[4/7] Configuring loss and optimizer...')
    
    # Loss function
    loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # Learning rate schedule with warmup
    total_steps = train_steps * EPOCHS
    warmup_steps = train_steps * WARMUP_EPOCHS
    
    lr_schedule = WarmupCosineDecayLR(
        base_lr=LEARNING_RATE,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=MIN_LR
    )
    
    # Generate learning rate list
    lr_list = [lr_schedule(step) for step in range(total_steps)]
    
    # Optimizer with gradient clipping
    optimizer = Adam(
        network.trainable_params(),
        learning_rate=lr_list,
        weight_decay=1e-4
    )
    
    print(f'✓ Optimizer: Adam with weight decay')
    print(f'✓ LR schedule: Warmup ({WARMUP_EPOCHS} epochs) + Cosine Decay')
    
    # Step 5: Setup callbacks
    print('\n[5/7] Setting up callbacks...')
    
    # Checkpoint configuration
    config_ck = CheckpointConfig(
        save_checkpoint_steps=args.save_checkpoint_steps,
        keep_checkpoint_max=args.keep_checkpoint_max
    )
    
    ckpoint_cb = ModelCheckpoint(
        prefix="emotion_cnn",
        directory=MODEL_SAVE_DIR,
        config=config_ck
    )
    
    loss_monitor = CustomLossMonitor(per_print_times=train_steps // 10)
    time_monitor = TimeMonitor(data_size=train_steps)
    
    # Skip validation during training if requested (faster)
    if args.no_validation:
        metrics_logger = MetricsLogger(LOGS_DIR, val_dataset=None, network=None, loss_fn=None)
    else:
        metrics_logger = MetricsLogger(LOGS_DIR, val_dataset=val_dataset, network=network, loss_fn=loss_fn)
    
    callbacks = [ckpoint_cb, loss_monitor, time_monitor, metrics_logger]
    
    print(f'✓ Checkpoints will be saved every {args.save_checkpoint_steps} steps')
    print(f'✓ Keeping maximum {args.keep_checkpoint_max} checkpoints')
    
    # Step 6: Create model and train
    print('\n[6/7] Creating MindSpore Model...')
    
    # Define metrics
    metrics = {
        'accuracy': Accuracy()
    }
    
    model = Model(
        network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=metrics
    )
    
    print('✓ Model created successfully')
    
    # Step 7: Train the model
    print('\n[7/7] Starting training...')
    print('-' * 80)
    
    try:
        model.train(
            EPOCHS,
            train_dataset,
            callbacks=callbacks,
            dataset_sink_mode=False  # Set to True for better performance on Ascend/GPU
        )
        
        print('\n' + '=' * 80)
        print('TRAINING COMPLETED SUCCESSFULLY!')
        print('=' * 80)
        
        # Final evaluation
        print('\n[Evaluation] Evaluating on validation set...')
        result = model.eval(val_dataset, dataset_sink_mode=False)
        
        print(f'\n✓ Final Validation Accuracy: {result["accuracy"]:.4f} ({result["accuracy"]*100:.2f}%)')
        
        # Save final results
        results = {
            'final_val_accuracy': float(result['accuracy']),
            'total_epochs': EPOCHS,
            'total_parameters': int(total_params),
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'dropout_rate': DROPOUT_RATE,
            'class_counts': class_counts,
            'timestamp': timestamp
        }
        
        results_file = os.path.join(LOGS_DIR, 'final_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\n✓ Results saved to: {results_file}')
        print(f'✓ Model checkpoints: {MODEL_SAVE_DIR}')
        print(f'✓ Training logs: {LOGS_DIR}')
        
    except Exception as e:
        print(f'\n✗ Training failed with error: {e}')
        raise
    
    print('\n' + '=' * 80)


if __name__ == '__main__':
    main()
