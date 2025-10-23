# src/dataset.py
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob

IMG_SIZE = 64  # small model; increase if you have compute

def read_image(path, img_size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return img

def load_from_folder(data_dir, img_size=IMG_SIZE):
    """
    Expects folder structure: data_dir/class_id/*.png
    Returns (images, labels, class_names)
    """
    class_dirs = sorted([d for d in glob(os.path.join(data_dir, "*")) if os.path.isdir(d)])
    images, labels = [], []
    class_names = [os.path.basename(d) for d in class_dirs]
    for label_idx, d in enumerate(class_dirs):
        files = [f for f in glob(os.path.join(d, "*")) if f.lower().endswith(('.ppm', '.png'))]
        for f in files:
            try:
                img = read_image(f, img_size)
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print("skip", f, e)
    X = np.array(images)
    y = np.array(labels, dtype=np.int32)
    return X, y, class_names

def get_train_val(data_dir, val_size=0.2, img_size=IMG_SIZE, random_state=42):
    X, y, class_names = load_from_folder(data_dir, img_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), class_names

def get_train_val_test(data_dir, val_size=0.2, test_size=0.1, img_size=IMG_SIZE, random_state=42):
    """
    Splits data into train, val, test datasets
    """
    X, y, class_names = load_from_folder(data_dir, img_size)

    # Split train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Split train and val
    val_relative = val_size / (1 - test_size)  # adjust validation size relative to remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=random_state
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names

def make_tf_dataset(X, y, batch_size=32, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    def _preprocess(img, label):
        img = tf.cast(img, tf.float32)
        return img, label

    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

