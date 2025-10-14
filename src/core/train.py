import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils

# --- CONFIG ---
DATA_DIR = 'data'
IMG_SIZE = 48
EMOTIONS = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# --- LOAD DATA ---
images = []
labels = []

for idx, emotion in enumerate(EMOTIONS):
    folder = os.path.join(DATA_DIR, emotion)
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        images.append(img)
        labels.append(idx)

X = np.array(images)[..., np.newaxis]
y = utils.to_categorical(labels, num_classes=len(EMOTIONS))

# --- SPLIT DATA ---
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=labels)

# --- BUILD MODEL ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(EMOTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- TRAIN ---
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

# --- SAVE KERAS MODEL ---
model.save('emotion_model.h5')

# --- CONVERT TO TFLITE ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Training complete. TFLite model saved as emotion_model.tflite")