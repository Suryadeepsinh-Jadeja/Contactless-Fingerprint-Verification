import os
import cv2
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Config
IMAGE_SIZE = (128, 128)
DATASET_PATH = '/Users/suryadeepsinhjadeja/Documents/Codes/TeachableMachine/dataset/'
EPOCHS = 10
BATCH_SIZE = 32
TARGET_USER = 'user1'  # 👈 change this to your enrolled user folder name

# 1. Load and preprocess images
def load_data():
    users = os.listdir(DATASET_PATH)
    data = {}

    for user in users:
        user_path = os.path.join(DATASET_PATH, user)
        if os.path.isdir(user_path):
            images = []
            for file in os.listdir(user_path):
                if not (file.endswith('.jpg') or file.endswith('.png')):
                    continue
                img_path = os.path.join(user_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ Skipping unreadable file: {img_path}")
                    continue
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                images.append(img)
            if len(images) >= 2:
                data[user] = images
            else:
                print(f"⚠️ Skipping user '{user}' due to insufficient images.")
    return data

# 2. Create positive and negative pairs
def create_pairs(data):
    positives = []
    negatives = []

    if TARGET_USER not in data:
        raise Exception(f"User '{TARGET_USER}' not found in dataset.")

    target_images = data[TARGET_USER]

    # Positive pairs (same user)
    for i in range(len(target_images) - 1):
        positives.append([target_images[i], target_images[i + 1]])

    # Negative pairs (different users)
    for user in data:
        if user != TARGET_USER:
            for img in data[user]:
                target_img = random.choice(target_images)
                negatives.append([target_img, img])

    pairs = np.array(positives + negatives)
    labels = np.array([1]*len(positives) + [0]*len(negatives))
    return pairs, labels

# 3. Build the Siamese Network
def build_siamese_model(input_shape):
    def build_base_network():
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Conv2D(32, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        return model

    base_network = build_base_network()

    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    feat_a = base_network(input_a)
    feat_b = base_network(input_b)

    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([feat_a, feat_b])
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = models.Model(inputs=[input_a, input_b], outputs=output)
    return model

# Main function
def train_model():
    data = load_data()
    pairs, labels = create_pairs(data)

    X1 = pairs[:, 0].reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    X2 = pairs[:, 1].reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    y = labels

    X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
        X1, X2, y, test_size=0.2, random_state=42)

    model = build_siamese_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit([X1_train, X2_train], y_train,
              validation_data=([X1_val, X2_val], y_val),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS)

    model.save("fingerprint_siamese_model.h5")
    print("✅ Model trained and saved as 'fingerprint_siamese_model.h5'")

# Run the training
train_model()
