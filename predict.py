# src/predict.py
import sys
import numpy as np
import cv2
import tensorflow as tf
from src.dataset import read_image, IMG_SIZE
import argparse

def predict(model_path, image_path, class_names_path=None):
    model = tf.keras.models.load_model(model_path)
    img = read_image(image_path, IMG_SIZE)
    img_batch = np.expand_dims(img, axis=0)
    preds = model.predict(img_batch)[0]
    top_idx = np.argmax(preds)
    print(f"Predicted class index: {top_idx}, confidence: {preds[top_idx]:.4f}")
    # if you want class names
    if class_names_path:
        import json
        with open(class_names_path, 'r') as f:
            classes = json.load(f)
        print("Predicted label:", classes[top_idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--classes", required=False, help="json file mapping indices to names")
    args = parser.parse_args()
    predict(args.model, args.image, args.classes)
