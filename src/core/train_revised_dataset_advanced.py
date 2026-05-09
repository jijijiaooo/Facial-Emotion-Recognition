#!/usr/bin/env python3
"""
Advanced CNN Training Script for Emotion Classification
Implements:
1. Enhanced data augmentation (contrast, noise, cutout)
2. Squeeze-and-Excitation (SE) channel attention after spatial attention
3. Option to use class-balanced loss (effective number of samples)
4. Stochastic Weight Averaging (SWA) and Mixup
5. F1-score and confusion matrix logging after training
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

# Enable mixed precision for Apple Silicon (M1/M2/M3/M4)
mixed_precision.set_global_policy('mixed_float16')

# Print policy for confirmation
print(f"Mixed precision policy: {mixed_precision.global_policy()}")
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- 1. Data Augmentation Utilities ---

# --- Batch-compatible Cutout Layer ---
class Cutout(layers.Layer):
    def __init__(self, mask_size=24, **kwargs):
        super().__init__(**kwargs)
        self.mask_size = mask_size
    def call(self, images):
        # images: (batch, h, w, c)
        batch_size = tf.shape(images)[0]
        h = tf.shape(images)[1]
        w = tf.shape(images)[2]
        c = tf.shape(images)[3]
        mask = tf.ones_like(images)
        for i in range(batch_size):
            y = tf.random.uniform([], 0, h, dtype=tf.int32)
            x = tf.random.uniform([], 0, w, dtype=tf.int32)
            y1 = tf.clip_by_value(y - self.mask_size // 2, 0, h)
            y2 = tf.clip_by_value(y + self.mask_size // 2, 0, h)
            x1 = tf.clip_by_value(x - self.mask_size // 2, 0, w)
            x2 = tf.clip_by_value(x + self.mask_size // 2, 0, w)
            mask_i = mask[i]
            paddings = [[y1, h - y2], [x1, w - x2], [0, 0]]
            cutout_area = tf.zeros([y2 - y1, x2 - x1, c], dtype=images.dtype)
            cutout_area = tf.pad(cutout_area, paddings, constant_values=1)
            mask_i = mask_i * cutout_area
            mask = tf.tensor_scatter_nd_update(mask, [[i]], [mask_i])
        return images * mask

# --- 2. Squeeze-and-Excitation Block ---
def se_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape([1,1,filters])(se)
    return layers.Multiply()([input_tensor, se])

# --- 3. Class-Balanced Loss ---
def get_class_balanced_weights(counts, beta=0.9999):
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(counts)
    return weights

class ClassBalancedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, gamma=2.0, label_smoothing=0.0, name='cb_focal_loss'):
        super().__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def call(self, y_true, y_pred):
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
            y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / num_classes)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        cb_weight = tf.reduce_sum(self.class_weights * y_true, axis=-1)
        loss = cb_weight * tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
        return loss
    def get_config(self):
        config = super().get_config()
        config.update({'class_weights': self.class_weights.numpy().tolist(), 'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return config

# --- 4. Mixup Utility ---
def mixup(batch_x, batch_y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = batch_x.shape[0]
    index = np.random.permutation(batch_size)
    mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
    mixed_y = lam * batch_y + (1 - lam) * batch_y[index]
    return mixed_x, mixed_y

# --- 5. Model Definition (with SE block) ---
def build_advanced_cnn(input_shape, num_classes, dropout_rate=0.48, l2_strength=0.00015):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = layers.RandomContrast(0.2)(x)
    x = layers.GaussianNoise(0.05)(x)
    # Conv blocks (as before)
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x1)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x2 = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x2)
    x = layers.Dropout(0.35)(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(448, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(448, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(448, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Conv2D(448, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(448, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # Spatial Attention
    attention = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    x = layers.Multiply()([x, attention])
    # Channel Attention (SE block)
    x = se_block(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(768, kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(384, kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(192, kernel_regularizer=l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate * 0.8)(x)
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_strength))(x)
    return models.Model(inputs, outputs)

# --- Main Training Loop ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/revised_dataset_balanced/train')
    parser.add_argument('--val_dir', default='data/revised_dataset_balanced/validation')
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--dropout', type=float, default=0.48)
    parser.add_argument('--l2_strength', type=float, default=0.00015)
    parser.add_argument('--logs_dir', default=None)
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--label_smoothing', type=float, default=0.06)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--swa', action='store_true')
    args = parser.parse_args()

    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    INITIAL_LR = args.lr
    DROPOUT_RATE = args.dropout
    L2_STRENGTH = args.l2_strength
    LABEL_SMOOTHING = args.label_smoothing
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.val_dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    MODEL_SAVE_PATH = os.path.join(args.model_dir, f'advanced_emotion_cnn_{timestamp}.keras')
    LOGS_DIR = args.logs_dir or f'logs/advanced_training_{timestamp}'
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Data generators
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        shear_range=0.08,
        zoom_range=0.12,
        horizontal_flip=True,
        brightness_range=[0.88, 1.12],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        color_mode='grayscale', class_mode='categorical', shuffle=True
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        color_mode='grayscale', class_mode='categorical', shuffle=False
    )
    class_counts = np.bincount(train_gen.classes)
    class_weights = get_class_balanced_weights(class_counts)

    # Print/save class indices
    print('Class indices:', train_gen.class_indices)
    with open(os.path.join(LOGS_DIR, 'class_indices.json'), 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # Model
    model = build_advanced_cnn((IMG_SIZE, IMG_SIZE, 1), len(train_gen.class_indices), dropout_rate=DROPOUT_RATE, l2_strength=L2_STRENGTH)
    loss_fn = ClassBalancedFocalLoss(class_weights, gamma=2.0, label_smoothing=LABEL_SMOOTHING)
    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Print/save model/optimizer config
    print('Model summary:')
    model.summary()
    print('Optimizer config:', optimizer.get_config())
    with open(os.path.join(LOGS_DIR, 'optimizer_config.json'), 'w') as f:
        json.dump(optimizer.get_config(), f, indent=2)

    # Print/save training arguments
    print('Training arguments:', vars(args))
    with open(os.path.join(LOGS_DIR, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Callbacks
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=LOGS_DIR, histogram_freq=1, write_graph=True, update_freq='epoch')
    ]
    if args.swa:
        try:
            from tensorflow.keras.callbacks import StochasticWeightAveraging
            callbacks.append(StochasticWeightAveraging())
        except ImportError:
            print('SWA not available in this TensorFlow version.')

    # --- Progress bar training using model.fit ---
    # Always apply Cutout; apply Mixup if enabled
    def custom_aug_gen(gen):
        while True:
            x_batch, y_batch = next(gen)
            if args.mixup:
                x_batch, y_batch = mixup(x_batch, y_batch)
            x_batch = Cutout()(x_batch)
            yield x_batch, y_batch

    train_input = custom_aug_gen(train_gen)

    print('\n[Training with Keras progress bar]')
    history = model.fit(
        train_input,
        steps_per_epoch=len(train_gen),
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f'Model saved: {MODEL_SAVE_PATH}')

    # Evaluation
    val_gen.reset()
    y_true = val_gen.classes
    y_pred = model.predict(val_gen, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_true, y_pred_labels, average='weighted')
    print(f'Weighted F1-score: {f1:.4f}')
    print('Classification Report:')
    print(classification_report(y_true, y_pred_labels, target_names=list(val_gen.class_indices.keys())))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred_labels))
    # Save history and final metrics
    with open(os.path.join(LOGS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history.history, f, indent=2)
    with open(os.path.join(LOGS_DIR, 'final_metrics.json'), 'w') as f:
        json.dump({
            'best_val_acc': float(np.max(history.history['val_accuracy'])),
            'best_val_loss': float(np.min(history.history['val_loss'])),
            'final_f1': float(f1)
        }, f, indent=2)

if __name__ == '__main__':
    main()
