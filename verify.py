import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Config
IMAGE_SIZE = (128, 128)
MODEL_PATH = "fingerprint_siamese_model.h5"

# Load model
model = load_model(MODEL_PATH, compile=False)

# Preprocess function
def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {path}")
        sys.exit(1)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    return img

# Main
if len(sys.argv) != 3:
    print("Usage: python verify.py <image1> <image2>")
    sys.exit(1)

img1_path = sys.argv[1]
img2_path = sys.argv[2]

img1 = preprocess_image(img1_path)
img2 = preprocess_image(img2_path)

# Predict
score = model.predict([img1, img2])[0][0]
print(f"Similarity Score: {score:.4f}")

if score > 0.5:
    print("✅ Match (same person)")
else:
    print("❌ Not a match (different person)")
