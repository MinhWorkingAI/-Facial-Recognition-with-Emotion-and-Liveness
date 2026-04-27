import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import os
from pathlib import Path

from data_explore import load_datasets, create_dataset_summary


# Load datasets
train_ds, val_ds, _ = load_datasets()

# Create dataset summary BEFORE cache/prefetch
summary_df = create_dataset_summary(train_ds, val_ds, _)
print(summary_df.to_string(index=False))

# Get class names BEFORE cache/prefetch
class_names = train_ds.class_names

def calculate_class_weights(real_count, spoof_count):
    total_count = real_count + spoof_count

    return {
        0: total_count / (2 * real_count),
        1: total_count / (2 * spoof_count)
    }

class_weight = calculate_class_weights(
    real_count=1223,
    spoof_count=7076
)

print("Class weights:", class_weight)
print("Classes:", class_names)

if len(class_names) != 2:
    raise ValueError("This model is for binary classification only. Expected 2 classes.")

# Convert labels to binary format for sigmoid + binary_crossentropy
def make_binary_labels(images, labels):
    labels = tf.cast(labels, tf.float32)
    labels = tf.expand_dims(labels, axis=-1)
    return images, labels


train_ds = train_ds.map(make_binary_labels)
val_ds = val_ds.map(make_binary_labels)


# Improve dataset loading performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Plot training and validation loss
def plot_loss(history, save_path=None):
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


# Data Augmentation
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15)
])


# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

#Reduce learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

L2 = regularizers.l2(0.0001)

# Build CNN from scratch
def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(160, 160, 3)),

        data_aug,
        layers.Rescaling(1./255),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu', kernel_regularizer=L2),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Binary output: real or spoof
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


# Train Model
cnn_model = build_cnn()
cnn_model.summary()

history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight
)


# Save model folder
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd()

model_dir = current_dir / "model"
model_dir.mkdir(parents=True, exist_ok=True)


# Save validation loss graph
loss_graph_path = model_dir / "cnn_validation_loss.png"
plot_loss(history, save_path=loss_graph_path)

print(f"Loss graph saved to: {loss_graph_path}")


# Save model
model_path = model_dir / "cnn_model.keras"
cnn_model.save(model_path)

print(f"Model saved to: {model_path}")