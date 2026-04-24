import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from anti_spoofing_module.data_paths import get_data_dirs
except ModuleNotFoundError:
    from data_paths import get_data_dirs

# Define Dataset Properties
# Image size for MobileNetV2 input; load train/val/test splits.
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

def get_class_counts(dataset):
    class_counts = {class_name: 0 for class_name in dataset.class_names}
    for path in dataset.file_paths:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name in class_counts:
            class_counts[class_name] += 1
    return class_counts

def main() -> None:
    train_dir, val_dir, test_dir = get_data_dirs()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
    )

    train_counts = get_class_counts(train_ds)
    val_counts = get_class_counts(val_ds)
    test_counts = get_class_counts(test_ds)

    summary_rows = []
    for split_name, dataset, class_counts in [
        ("Train", train_ds, train_counts),
        ("Validation", val_ds, val_counts),
        ("Test", test_ds, test_counts),
    ]:
        row = {
            "Split": split_name,
            "Total Files": len(dataset.file_paths),
            "Num Classes": len(dataset.class_names),
        }
        for class_name in dataset.class_names:
            row[f"{class_name} Count"] = class_counts.get(class_name, 0)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    datasets = [("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)]
    cols = min(2, max(len(ds.class_names) for _, ds in datasets))
    fig, axes = plt.subplots(len(datasets), cols, figsize=(8, 12))
    axes = np.atleast_2d(axes)

    for row, (split_name, dataset) in enumerate(datasets):
        for col, class_name in enumerate(dataset.class_names[:cols]):
            class_paths = [
                path
                for path in dataset.file_paths
                if os.path.basename(os.path.dirname(path)) == class_name
            ]
            ax = axes[row, col]
            if class_paths:
                image_path = np.random.choice(class_paths)
                image = plt.imread(image_path)
                ax.imshow(image, cmap="gray")
                ax.set_title(f"{split_name} | {class_name}")
            else:
                ax.set_title(f"{split_name} | {class_name} (no image)")
            ax.axis("off")

    plt.tight_layout()
    output_path = Path(__file__).resolve().parent / "data_preview.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved image: {output_path}")


if __name__ == "__main__":
    main()