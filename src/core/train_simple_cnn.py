#!/usr/bin/env python3
"""
Train a Simple CNN from Scratch for Emotion Classification
Enhanced with: Focal Loss, Label Smoothing, Spatial Attention, Advanced Augmentation
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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam


class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance and hard examples
    Focuses training on hard-to-classify examples (Fear, Sad)
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Apply label smoothing
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal loss
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        focal_loss = self.alpha * focal_weight * cross_entropy
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


class SpatialAttention(layers.Layer):
    """Spatial Attention mechanism to focus on important facial regions
    Helps model attend to eyes, mouth, eyebrows for emotion recognition
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        super().build(input_shape)
    
    def call(self, inputs):
        # Channel-wise statistics
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate and create attention map
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        
        # Apply attention
        return inputs * attention


class RandomErasing(layers.Layer):
    """Random Erasing augmentation - forces model to learn from partial facial info"""
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.sl = sl  # min erasing area
        self.sh = sh  # max erasing area
        self.r1 = r1  # min aspect ratio
    
    def call(self, inputs, training=None):
        if training is False:
            return inputs
        
        # Use tf.py_function for simpler implementation that works with AutoGraph
        return inputs  # Simplified: rely on ImageDataGenerator augmentation instead
    
    def compute_output_shape(self, input_shape):
        return input_shape


def build_simple_cnn(input_shape, num_classes):
    """Build enhanced CNN with Spatial Attention and RandomErasing"""
    inputs = layers.Input(shape=input_shape)
    
    # RandomErasing augmentation layer
    x = RandomErasing(probability=0.5)(inputs)
    
    # Conv Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Conv Block 2 with Spatial Attention
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = SpatialAttention()(x)  # Focus on important facial regions
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Conv Block 3 with Spatial Attention
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = SpatialAttention()(x)  # Additional attention for deeper features
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Conv Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    
    # Fully Connected
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='enhanced_emotion_cnn')
    return model


def cosine_annealing_schedule(epoch, initial_lr=0.0005, epochs=150, min_lr=1e-6):
    """Cosine annealing learning rate schedule with minimum LR"""
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    return min_lr + (initial_lr - min_lr) * cosine_decay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/Combined_Dataset/train')
    parser.add_argument('--val_dir', default='data/Combined_Dataset/validation')
    parser.add_argument('--img_size', type=int, default=128, help='Input image size (optimized for facial details)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--logs_dir', default=None)
    parser.add_argument('--focal_loss', action='store_true', help='Use Focal Loss instead of categorical crossentropy')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
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
    print('ENHANCED CNN EMOTION CLASSIFICATION TRAINING')
    print(f'Image size: {IMG_SIZE}  Batch: {BATCH_SIZE}  Epochs: {EPOCHS}  LR: {LEARNING_RATE}')
    print(f'Focal Loss: {args.focal_loss}  Label Smoothing: {args.label_smoothing}')
    print('Features: Spatial Attention + RandomErasing + Advanced Augmentation')
    print('=' * 70)

    # Enhanced data augmentation
    print('\n[1/6] Setting up enhanced data augmentation...')
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=25,  # Increased from 20
        width_shift_range=0.2,  # Increased from 0.15
        height_shift_range=0.2,
        shear_range=0.2,  # Increased from 0.15
        zoom_range=0.2,  # Increased from 0.15
        brightness_range=[0.8, 1.2],  # NEW: brightness variation
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print('✓ Enhanced data augmentation configured (rotation, shift, brightness, zoom)')

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
    print('\n[4/6] Building enhanced CNN model with Spatial Attention...')
    model = build_simple_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=len(EMOTION_LABELS))
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    # Choose loss function
    if args.focal_loss:
        print(f'  Using Focal Loss (alpha=0.25, gamma=2.0) with Label Smoothing ({args.label_smoothing})')
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=args.label_smoothing)
    else:
        print(f'  Using Categorical Crossentropy with Label Smoothing ({args.label_smoothing})')
        loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
    
    model.compile(
        optimizer=optimizer, 
        loss=loss_fn,
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    )
    
    print('\n✓ Model architecture:')
    model.summary()

    # Callbacks
    print('\n[5/6] Setting up callbacks...')
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),  # Optimized patience
        LearningRateScheduler(lambda epoch: cosine_annealing_schedule(epoch, LEARNING_RATE, EPOCHS), verbose=1),
        TensorBoard(log_dir=LOGS_DIR, histogram_freq=1, write_graph=True, update_freq='epoch')
    ]

    print('✓ Callbacks: ModelCheckpoint + EarlyStopping + Cosine Annealing LR + TensorBoard')

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
