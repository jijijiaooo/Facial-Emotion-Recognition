#!/usr/bin/env python3
"""
Train Enhanced CNN on Combined Dataset for Emotion Classification
Optimized for large-scale training with ~39K images across 7 emotion classes
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    ReduceLROnPlateau, 
    TensorBoard,
    LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_improved_cnn(input_shape, num_classes, dropout_rate=0.5):
    """
    Build an improved CNN architecture optimized for large datasets
    
    Improvements:
    - Deeper architecture to leverage more training data
    - Residual-like skip connections for better gradient flow
    - Adaptive pooling strategy
    - L2 regularization to prevent overfitting with large dataset
    - Optimized dropout rates
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Conv Block
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Conv Block 2 with increased capacity
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv Block 3 - deeper feature extraction
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)
    
    # Conv Block 4 - high-level features
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Global Average Pooling for spatial invariance
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with strong regularization
    x = layers.Dense(512, kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(256, kernel_regularizer=l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0001))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def cosine_decay_with_warmup(epoch, total_epochs=100, warmup_epochs=5, 
                              initial_lr=0.001, min_lr=1e-6):
    """
    Learning rate schedule with warmup and cosine decay
    Works better for large datasets
    """
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def parse_args():
    parser = argparse.ArgumentParser(description='Train CNN on Combined Emotion Dataset')
    parser.add_argument('--train_dir', default='data/combined_dataset_complete/train',
                        help='Training data directory')
    parser.add_argument('--val_dir', default='data/combined_dataset_complete/validation',
                        help='Validation data directory')
    parser.add_argument('--img_size', type=int, default=112, 
                        help='Input image size (increased for larger dataset)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for dense layers')
    parser.add_argument('--model_dir', default='models',
                        help='Directory to save trained models')
    parser.add_argument('--logs_dir', default=None,
                        help='Directory for TensorBoard logs')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training for faster training')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs for learning rate')
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    INITIAL_LR = args.lr
    DROPOUT_RATE = args.dropout
    WARMUP_EPOCHS = args.warmup_epochs

    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    MODEL_SAVE_PATH = os.path.join(args.model_dir, f'emotion_combined_cnn_{timestamp}.h5')
    LOGS_DIR = args.logs_dir or f'logs/training_combined_{timestamp}'

    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Enable mixed precision if requested (speeds up training on modern GPUs)
    if args.use_mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print('✓ Mixed precision training enabled')

    print('=' * 80)
    print('COMBINED DATASET CNN TRAINING')
    print('=' * 80)
    print(f'Image size: {IMG_SIZE}x{IMG_SIZE}')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'Epochs: {EPOCHS}')
    print(f'Initial LR: {INITIAL_LR}')
    print(f'Dropout: {DROPOUT_RATE}')
    print(f'Warmup epochs: {WARMUP_EPOCHS}')
    print('=' * 80)

    # Enhanced data augmentation for larger dataset
    print('\n[1/7] Setting up data augmentation...')
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=25,          # Increased for more variety
        width_shift_range=0.2,      # Increased
        height_shift_range=0.2,     # Increased
        shear_range=0.2,            # Increased
        zoom_range=0.2,             # Increased
        horizontal_flip=True,
        brightness_range=[0.8, 1.2], # Add brightness variation
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    print('✓ Data augmentation configured with enhanced parameters')

    # Load data
    print('\n[2/7] Loading data...')
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        classes=EMOTION_LABELS,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical',
        classes=EMOTION_LABELS,
        shuffle=False
    )

    print(f'✓ Training samples: {train_generator.samples:,}')
    print(f'✓ Validation samples: {val_generator.samples:,}')
    print(f'✓ Classes: {EMOTION_LABELS}')

    # Compute class weights to handle imbalance
    print('\n[3/7] Computing class weights...')
    labels = train_generator.classes
    counts = np.bincount(labels, minlength=len(EMOTION_LABELS))
    total = labels.shape[0]
    class_weight = {i: float(total) / (len(EMOTION_LABELS) * max(1, counts[i])) 
                    for i in range(len(EMOTION_LABELS))}
    
    print('  Class distribution:')
    for i, (label, count) in enumerate(zip(EMOTION_LABELS, counts)):
        print(f'    {label:10s}: {count:6d} samples (weight: {class_weight[i]:.3f})')
    
    # Build improved model
    print('\n[4/7] Building improved CNN model...')
    model = build_improved_cnn(
        input_shape=(IMG_SIZE, IMG_SIZE, 1), 
        num_classes=len(EMOTION_LABELS),
        dropout_rate=DROPOUT_RATE
    )
    
    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    print('\n✓ Model architecture:')
    model.summary()
    
    total_params = model.count_params()
    print(f'\n✓ Total parameters: {total_params:,}')

    # Setup callbacks
    print('\n[5/7] Setting up callbacks...')
    
    # Learning rate scheduler with warmup
    lr_scheduler = LearningRateScheduler(
        lambda epoch: cosine_decay_with_warmup(
            epoch, 
            total_epochs=EPOCHS, 
            warmup_epochs=WARMUP_EPOCHS,
            initial_lr=INITIAL_LR,
            min_lr=1e-6
        ),
        verbose=1
    )
    
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max', 
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss', 
            patience=20,  # Increased patience for larger dataset
            restore_best_weights=True, 
            verbose=1
        ),
        lr_scheduler,  # Use cosine decay instead of ReduceLROnPlateau
        TensorBoard(
            log_dir=LOGS_DIR, 
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    print('✓ Callbacks configured:')
    print('  - ModelCheckpoint: saving best model by val_accuracy')
    print('  - EarlyStopping: patience=20 on val_loss')
    print('  - LR Scheduler: cosine decay with warmup')
    print('  - TensorBoard: logging to', LOGS_DIR)

    # Calculate steps
    print('\n[6/7] Preparing training...')
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    val_steps = val_generator.samples // BATCH_SIZE
    
    print(f'✓ Steps per epoch: {steps_per_epoch}')
    print(f'✓ Validation steps: {val_steps}')

    # Train
    print('\n[7/7] Training...')
    print('=' * 80)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    # Final evaluation
    print('\n' + '=' * 80)
    print('FINAL EVALUATION')
    print('=' * 80)
    
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f'Final Validation Loss:     {val_loss:.4f}')
    print(f'Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)')

    # Save training history
    history_file = os.path.join(LOGS_DIR, 'training_history.json')
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    
    # Add metadata
    metadata = {
        'timestamp': timestamp,
        'dataset': 'combined_dataset_complete',
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(history.history['loss']),
        'total_epochs': EPOCHS,
        'initial_lr': INITIAL_LR,
        'dropout_rate': DROPOUT_RATE,
        'warmup_epochs': WARMUP_EPOCHS,
        'train_samples': train_generator.samples,
        'val_samples': val_generator.samples,
        'total_params': total_params,
        'final_val_accuracy': float(val_accuracy),
        'final_val_loss': float(val_loss),
        'class_distribution': {label: int(count) for label, count in zip(EMOTION_LABELS, counts)},
        'class_weights': {label: float(class_weight[i]) for i, label in enumerate(EMOTION_LABELS)}
    }
    
    full_history = {
        'metadata': metadata,
        'history': history_dict
    }
    
    with open(history_file, 'w') as f:
        json.dump(full_history, f, indent=2)
    
    print(f'\n✓ Training history saved: {history_file}')

    # Summary
    print('\n' + '=' * 80)
    print('TRAINING COMPLETED!')
    print('=' * 80)
    print(f'Model saved:       {MODEL_SAVE_PATH}')
    print(f'Logs directory:    {LOGS_DIR}')
    print(f'Final val acc:     {val_accuracy*100:.2f}%')
    print(f'Total parameters:  {total_params:,}')
    print(f'Training samples:  {train_generator.samples:,}')
    print('=' * 80)
    
    # Print best epoch info
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    print(f'\nBest epoch: {best_epoch} with val_accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)')
    
    print('\nTo visualize training:')
    print(f'  tensorboard --logdir={LOGS_DIR}')


if __name__ == '__main__':
    main()
