import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report


try:
    from anti_spoofing_module.data_paths import get_data_dirs
except ModuleNotFoundError:
    from data_paths import get_data_dirs



IMG_SIZE = (160, 160)
BATCH_SIZE = 32

_, _, test_dir = get_data_dirs()

print("Test directory:", test_dir)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
print("Classes:", class_names)

if len(class_names) != 2:
    raise ValueError("This test script is for binary classification only. Expected 2 classes.")

if class_names != ["real", "spoof"]:
    raise ValueError(f"Unexpected class order: {class_names}")


# Convert labels to binary format for sigmoid model
def make_binary_labels(images, labels):
    labels = tf.cast(labels, tf.float32)
    labels = tf.expand_dims(labels, axis=-1)
    return images, labels


test_ds = test_ds.map(make_binary_labels)

AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Load Saved Model

try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd()

model_path = current_dir / "model" / "cnn_model.keras"

print("Loading model from:", model_path)

model = tf.keras.models.load_model(model_path)

#Evaluate Model
test_results = model.evaluate(test_ds, return_dict=True)

print("\nTest Results:")
for metric_name, metric_value in test_results.items():
    print(f"{metric_name}: {metric_value:.4f}")

#Predict

y_true = []
y_pred_probs = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten())
    y_pred_probs.extend(preds.flatten())

y_true = np.array(y_true).astype(int)
y_pred_probs = np.array(y_pred_probs)

# Since class order is ['real', 'spoof']:
# prediction < 0.5  = real
# prediction >= 0.5 = spoof
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    y_pred = (y_pred_probs >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    ))

    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

#Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

#Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

print("\nConfusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm)

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar(im)
plt.tight_layout()

confusion_matrix_path = current_dir / "model" / "confusion_matrix.png"
plt.savefig(confusion_matrix_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Confusion matrix saved to: {confusion_matrix_path}")