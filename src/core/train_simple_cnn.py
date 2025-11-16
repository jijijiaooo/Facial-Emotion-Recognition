#!/usr/bin/env python3
"""
Train a Simple CNN from Scratch for Emotion Classification
This approach often works better than transfer learning for emotion recognition
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam

def build_simple_cnn(input_shape, num_classes):
    """Build a simple but effective CNN for emotion recognition"""
    model = models.Sequential([
        # Conv Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/Emotion_Classification/train')
    parser.add_argument('--val_dir', default='data/Emotion_Classification/validation')
    parser.add_argument('--img_size', type=int, default=96, help='Input image size (square)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--logs_dir', default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    MODEL_SAVE_PATH = os.path.join(args.model_dir, f'emotion_simple_cnn_{timestamp}.h5')
    LOGS_DIR = args.logs_dir or f'logs/training_{timestamp}'

    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    print('=' * 70)
    print('SIMPLE CNN EMOTION CLASSIFICATION TRAINING')
    print(f'Image size: {IMG_SIZE}  Batch: {BATCH_SIZE}  Epochs: {EPOCHS}  LR: {LEARNING_RATE}')
    print('=' * 70)

    # Data augmentation
    print('\n[1/6] Setting up data augmentation...')
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print('✓ Data augmentation configured')

    # Generators - use grayscale for simpler learning
    print('\n[2/6] Loading data...')
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',  # Simpler features
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

    print(f'✓ Training: {train_generator.samples}  Validation: {val_generator.samples}')

    # Class weights
    print('\n[3/6] Computing class weights...')
    labels = train_generator.classes
    counts = np.bincount(labels, minlength=len(EMOTION_LABELS))
    total = labels.shape[0]
    class_weight = {i: float(total) / (len(EMOTION_LABELS) * max(1, counts[i])) for i in range(len(EMOTION_LABELS))}
    print(f'  Class counts: {counts.tolist()}')
    print(f'  Class weights: {class_weight}')

    # Build model
    print('\n[4/6] Building CNN model...')
    model = build_simple_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(EMOTION_LABELS))
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('\n✓ Model architecture:')
    model.summary()

    # Callbacks
    print('\n[5/6] Setting up callbacks...')
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=LOGS_DIR, histogram_freq=1)
    ]

    print('✓ Callbacks configured')

    # Train
    print('\n[6/6] Training...')
    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    val_steps = max(1, val_generator.samples // BATCH_SIZE)

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

    # Evaluate
    print('\n[7/7] Final evaluation...')
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f'✓ Final Validation Loss: {val_loss:.4f}')
    print(f'✓ Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)')

    # Save history
    history_file = os.path.join(LOGS_DIR, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    print(f'✓ Training history saved: {history_file}')

    print('\n' + '=' * 70)
    print('TRAINING COMPLETED!')
    print('=' * 70)
    print(f'Model: {MODEL_SAVE_PATH}')
    print(f'Logs: {LOGS_DIR}')
    print(f'Val accuracy: {val_accuracy*100:.2f}%')


if __name__ == '__main__':
    main()
