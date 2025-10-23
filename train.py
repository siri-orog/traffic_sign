# src/train.py
import os
import argparse
from src.dataset import get_train_val_test, make_tf_dataset, IMG_SIZE
from src.model import create_model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def main(args):
    train_dir = args.train_dir
    batch_size = args.batch_size
    epochs = args.epochs
    model_out = args.model_out

    # Split dataset into train, val, and test
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = get_train_val_test(
        train_dir, val_size=0.2, test_size=0.1, img_size=IMG_SIZE
    )
    num_classes = len(class_names)
    print("Classes:", num_classes)

    # Create TensorFlow datasets
    train_ds = make_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = make_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_ds = make_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)

    # Build and compile model
    model = create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(model_out, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
        ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-6),
    ]

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Save final model
    model.save(model_out.replace('.h5', '_final.h5'))

    # Evaluate on test dataset
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model_out", type=str, default="models/traffic_sign.h5")
    args = parser.parse_args()
    main(args)
