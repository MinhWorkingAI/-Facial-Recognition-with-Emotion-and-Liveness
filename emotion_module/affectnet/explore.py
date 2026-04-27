import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets

TRAIN_DIR = "AffectNet/Train"
TEST_DIR  = "AffectNet/Test"

train_dataset = datasets.ImageFolder(TRAIN_DIR)
test_dataset  = datasets.ImageFolder(TEST_DIR)

emotions = train_dataset.classes
print(f"Classes: {emotions}")

# Count per class
train_counts = [0] * len(emotions)
for _, label in train_dataset.samples:
    train_counts[label] += 1

test_counts = [0] * len(emotions)
for _, label in test_dataset.samples:
    test_counts[label] += 1

print(f"\n{'Emotion':<12} {'Train':>8} {'Test':>8}")
print("-" * 30)
for i, emotion in enumerate(emotions):
    print(f"{emotion:<12} {train_counts[i]:>8} {test_counts[i]:>8}")
print(f"\nTotal train: {sum(train_counts)}")
print(f"Total test:  {sum(test_counts)}")

# Plot distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(emotions, train_counts, color='steelblue')
axes[0].set_title("Train Distribution")
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(emotions, test_counts, color='coral')
axes[1].set_title("Test Distribution")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("explore/distribution.png")
print("Saved: explore/distribution.png")

# Preview samples
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

preview_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])
train_dataset.transform = preview_transform
loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
images, labels = next(iter(loader))

plt.figure(figsize=(14, 6))
for i in range(10):
    img = images[i].permute(1, 2, 0).numpy()
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(emotions[labels[i]])
    plt.axis("off")
plt.tight_layout()
plt.savefig("explore/samples.png")
print("Saved: explore/samples.png")