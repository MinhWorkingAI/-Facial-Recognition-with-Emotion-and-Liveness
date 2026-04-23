import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
# ── Config ───────────────────────────────────────────────────────────────
IMG_SIZE   = 96
BATCH_SIZE = 32
DATA_DIR   = "fer2013"
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
TEST_DIR   = os.path.join(DATA_DIR, "test")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── 1. Explore class distribution ─────────────────────────────────────────
def explore_dataset():
    print("=== Dataset Distribution ===")
    train_counts = {}
    test_counts  = {}

    for emotion in EMOTIONS:
        train_path = os.path.join(TRAIN_DIR, emotion)
        test_path  = os.path.join(TEST_DIR,  emotion)
        train_counts[emotion] = len(os.listdir(train_path)) if os.path.exists(train_path) else 0
        test_counts[emotion]  = len(os.listdir(test_path))  if os.path.exists(test_path)  else 0

    print(f"\n{'Emotion':<12} {'Train':>8} {'Test':>8}")
    print("-" * 30)
    for emotion in EMOTIONS:
        print(f"{emotion:<12} {train_counts[emotion]:>8} {test_counts[emotion]:>8}")
    print(f"\nTotal train: {sum(train_counts.values())}")
    print(f"Total test:  {sum(test_counts.values())}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(train_counts.keys(), train_counts.values(), color='steelblue')
    axes[0].set_title("Train Set Distribution")
    axes[0].set_xlabel("Emotion")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(test_counts.keys(), test_counts.values(), color='coral')
    axes[1].set_title("Test Set Distribution")
    axes[1].set_xlabel("Emotion")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()
    print("Saved: class_distribution.png")

    return train_counts

# ── 2. Compute class weights ───────────────────────────────────────────────
def compute_class_weights(train_counts):
    total     = sum(train_counts.values())
    n_classes = len(train_counts)
    class_weights = {
        i: total / (n_classes * count)
        for i, (_, count) in enumerate(train_counts.items())
    }
    print("\n=== Class Weights ===")
    for i, emotion in enumerate(EMOTIONS):
        print(f"  {emotion:<12}: {class_weights[i]:.3f}")
    return class_weights

# ── 3. Define transforms ───────────────────────────────────────────────────
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),  # FER2013 is grayscale, MobileNet expects 3ch
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats (for transfer learning)
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

# ── 4. Build datasets & dataloaders ───────────────────────────────────────
def build_dataloaders(train_transform, test_transform):
    full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset       = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

    # 90/10 train-val split
    val_size   = int(0.1 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=__import__('torch').Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n=== Dataloaders Ready ===")
    print(f"Train samples:      {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Class mapping:      {full_train_dataset.class_to_idx}")

    return train_loader, val_loader, test_loader, full_train_dataset.class_to_idx

# ── 5. Preview samples ─────────────────────────────────────────────────────
def preview_samples(train_loader, class_to_idx):
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    images, labels = next(iter(train_loader))

    # Denormalize for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    plt.figure(figsize=(14, 6))
    for i in range(min(10, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = std * img + mean          # undo normalization
        img = np.clip(img, 0, 1)

        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(idx_to_class[labels[i].item()])
        plt.axis("off")

    plt.suptitle("Sample Training Images (after augmentation)")
    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.show()
    print("Saved: sample_images.png")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_counts                              = explore_dataset()
    class_weights                             = compute_class_weights(train_counts)
    train_transform, test_transform           = get_transforms()
    train_loader, val_loader, test_loader, \
        class_to_idx                          = build_dataloaders(train_transform, test_transform)
    preview_samples(train_loader, class_to_idx)

    print("\n✅ Preprocessing done. Ready for model training.")