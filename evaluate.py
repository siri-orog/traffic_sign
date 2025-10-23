# src/evaluate.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from src.dataset import get_train_val_test, make_tf_dataset, IMG_SIZE
from src.model import create_model
import argparse

def evaluate(model_path, train_dir):
    # Split dataset to get test data
    _, _, (X_test, y_test), class_names = get_train_val_test(
        train_dir, val_size=0.2, test_size=0.1, img_size=IMG_SIZE
    )

    test_ds = make_tf_dataset(X_test, y_test, batch_size=32, shuffle=False)

    model = tf.keras.models.load_model(model_path)

    # Evaluate using the test dataset
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # Get predictions for classification report
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/traffic_sign.h5")
    parser.add_argument("--train_dir", type=str, default="data/train")  # <-- Use train_dir to split test
    args = parser.parse_args()
    evaluate(args.model_path, args.train_dir)

