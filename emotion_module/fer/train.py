import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import build_model, unfreeze_backbone, count_params
from preprocessing import (
    explore_dataset, compute_class_weights,
    get_transforms, build_dataloaders, EMOTIONS
)

# ── Config ────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS_FROZEN = 15    # train with frozen backbone first
EPOCHS_TUNED  = 15    # then fine-tune with partial unfreeze
LR_FROZEN     = 1e-3
LR_TUNED      = 1e-4  # lower LR when fine-tuning
SAVE_PATH     = "best_model.pth"

print(f"Using device: {DEVICE}")

# ── Training loop (one epoch) ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ── Validation loop ───────────────────────────────────────────────────────
def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ── Full training phase ───────────────────────────────────────────────────
def run_training(model, train_loader, val_loader, optimizer, criterion,
                 scheduler, epochs, phase_name):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n{'='*50}")
    print(f"  Phase: {phase_name}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            saved = "✅ saved"
        else:
            saved = ""

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} {saved}")

    print(f"\nBest val acc ({phase_name}): {best_val_acc:.4f}")
    return history


# ── Plot training curves ──────────────────────────────────────────────────
def plot_history(history1, history2, save_name="training_curves.png"):
    # Combine both phases
    train_loss = history1["train_loss"] + history2["train_loss"]
    val_loss   = history1["val_loss"]   + history2["val_loss"]
    train_acc  = history1["train_acc"]  + history2["train_acc"]
    val_acc    = history1["val_acc"]    + history2["val_acc"]
    epochs     = range(1, len(train_loss) + 1)
    split      = len(history1["train_loss"])  # where phase 2 starts

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss,   label="Val Loss")
    axes[0].axvline(x=split, color='gray', linestyle='--', label="Fine-tune starts")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, train_acc, label="Train Acc")
    axes[1].plot(epochs, val_acc,   label="Val Acc")
    axes[1].axvline(x=split, color='gray', linestyle='--', label="Fine-tune starts")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()
    print(f"Saved: {save_name}")


# ── Evaluate on test set ──────────────────────────────────────────────────
def evaluate_test(model, test_loader):
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            preds  = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=EMOTIONS))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Saved: confusion_matrix.png")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    train_counts                    = explore_dataset()
    class_weights                   = compute_class_weights(train_counts)
    train_transform, test_transform = get_transforms()
    train_loader, val_loader, \
        test_loader, class_to_idx   = build_dataloaders(train_transform, test_transform)

    # 2. Class weights tensor for loss function
    weights_tensor = torch.tensor(
        [class_weights[i] for i in range(len(EMOTIONS))],
        dtype=torch.float
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # ── Phase 1: Train head only (backbone frozen) ──────────────────────
    model = build_model(freeze_backbone=True).to(DEVICE)
    count_params(model)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FROZEN)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


    history1 = run_training(model, train_loader, val_loader,
                            optimizer, criterion, scheduler,
                            EPOCHS_FROZEN, "Phase 1 — Frozen Backbone")

    # ── Phase 2: Fine-tune (partial unfreeze) ───────────────────────────
    model = unfreeze_backbone(model, unfreeze_from_layer=14)
    count_params(model)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_TUNED)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    history2 = run_training(model, train_loader, val_loader,
                            optimizer, criterion, scheduler,
                            EPOCHS_TUNED, "Phase 2 — Fine-tuning")

    # ── Plot & evaluate ─────────────────────────────────────────────────
    plot_history(history1, history2)
    evaluate_test(model, test_loader)

    print("\n✅ Training complete. Model saved to:", SAVE_PATH)