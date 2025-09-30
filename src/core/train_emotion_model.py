import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers

# Path to your training data
train_dir = 'data/raf_db/processed/train'
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load images and labels
X, y = [], []
for idx, emotion in enumerate(emotion_labels):
    emotion_folder = os.path.join(train_dir, emotion)
    count = 0
    for img_name in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (48, 48))
            X.append(img)
            y.append(idx)
            count += 1
    print(f"Loaded {count} images for emotion: {emotion}")

print(f"Total images loaded: {len(X)}")

X = np.array(X).astype('float32') / 255.0
X = np.expand_dims(X, -1)
y = keras.utils.to_categorical(y, num_classes=len(emotion_labels))

# Build a simple CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(emotion_labels), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting model training...")
# Train the model
model.fit(X, y, epochs=20, batch_size=64, validation_split=0.1)

# Save the model
os.makedirs('models', exist_ok=True)
model.save('models/raf_db_simple_cnn.h5')
print("Model saved to models/raf_db_simple_cnn.h5")