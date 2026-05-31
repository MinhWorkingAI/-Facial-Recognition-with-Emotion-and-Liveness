from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import build_dataset
from .model import NormalizedEmbeddingModel, resnet18_face
from .triplet_data import TripletFaceDataset


try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Triplet Loss ResNet18 checkpoint for face verification."
    )
    parser.add_argument("--checkpoint", required=True, help="Triplet checkpoint, usually best.pth.")
    parser.add_argument(
        "--metadata",
        default="",
        help="Training metadata JSON. Defaults to training_metadata.json beside the checkpoint.",
    )
    parser.add_argument(
        "--history",
        default="",
        help="Training history JSON. Defaults to training_history.json beside the checkpoint.",
    )
    parser.add_argument("--data-root", default="", help="Override the dataset path stored in metadata.")
    parser.add_argument("--output-dir", default="", help="Defaults to <checkpoint_dir>_test.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override validation batch size.")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--max-positive-pairs", type=int, default=20000)
    parser.add_argument("--max-negative-pairs", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def progress(iterable, description: str, disabled: bool):
    if disabled or tqdm is None:
        return iterable
    return tqdm(iterable, desc=description, dynamic_ncols=True)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    no_progress: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings = []
    labels = []
    for images, batch_labels in progress(loader, "test embeddings", no_progress):
        embeddings.append(model(images.to(device, non_blocking=True)).cpu())
        labels.append(batch_labels.long())
    return torch.cat(embeddings), torch.cat(labels)


@torch.no_grad()
def evaluate_triplets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    no_progress: bool,
) -> dict[str, float]:
    model.eval()
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    total_loss = 0.0
    total_positive_distance = 0.0
    total_negative_distance = 0.0
    total_correct = 0
    total_active = 0
    total_seen = 0

    for anchor, positive, negative, _, _ in progress(loader, "test triplets", no_progress):
        anchor = anchor.to(device, non_blocking=True)
        positive = positive.to(device, non_blocking=True)
        negative = negative.to(device, non_blocking=True)
        batch_size = anchor.size(0)
        embeddings = model(torch.cat([anchor, positive, negative], dim=0))
        anchor_embedding = embeddings[:batch_size]
        positive_embedding = embeddings[batch_size:2 * batch_size]
        negative_embedding = embeddings[2 * batch_size:]
        loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
        positive_distance = F.pairwise_distance(anchor_embedding, positive_embedding)
        negative_distance = F.pairwise_distance(anchor_embedding, negative_embedding)

        total_loss += loss.item() * batch_size
        total_positive_distance += positive_distance.sum().item()
        total_negative_distance += negative_distance.sum().item()
        total_correct += (positive_distance + margin < negative_distance).sum().item()
        total_active += (positive_distance - negative_distance + margin > 0).sum().item()
        total_seen += batch_size

    return {
        "triplet_loss": total_loss / max(total_seen, 1),
        "triplet_accuracy": total_correct / max(total_seen, 1),
        "positive_distance": total_positive_distance / max(total_seen, 1),
        "negative_distance": total_negative_distance / max(total_seen, 1),
        "active_triplet_ratio": total_active / max(total_seen, 1),
        "triplets": total_seen,
    }


def build_verification_pairs(
    labels: torch.Tensor,
    negative_ratio: int,
    max_positive_pairs: int,
    max_negative_pairs: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    labels_by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        labels_by_class[int(label)].append(idx)

    positive_pairs = [
        (left_idx, right_idx)
        for indices in labels_by_class.values()
        for left_pos, left_idx in enumerate(indices)
        for right_idx in indices[left_pos + 1:]
    ]
    rng = random.Random(seed)
    if max_positive_pairs > 0 and len(positive_pairs) > max_positive_pairs:
        positive_pairs = rng.sample(positive_pairs, max_positive_pairs)

    num_items = labels.numel()
    all_pairs = num_items * (num_items - 1) // 2
    positive_total = sum(len(indices) * (len(indices) - 1) // 2 for indices in labels_by_class.values())
    negative_total = all_pairs - positive_total
    negative_target = min(negative_total, len(positive_pairs) * max(negative_ratio, 0))
    if max_negative_pairs > 0:
        negative_target = min(negative_target, max_negative_pairs)

    negative_pairs: set[tuple[int, int]] = set()
    while len(negative_pairs) < negative_target:
        left_idx, right_idx = rng.sample(range(num_items), 2)
        if labels[left_idx] == labels[right_idx]:
            continue
        negative_pairs.add(tuple(sorted((left_idx, right_idx))))

    return [(left, right, 1) for left, right in positive_pairs] + [
        (left, right, 0) for left, right in sorted(negative_pairs)
    ]


def score_pairs(embeddings: torch.Tensor, pairs: Sequence[tuple[int, int, int]]) -> torch.Tensor:
    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    return (embeddings[left] * embeddings[right]).sum(dim=1)


def compute_roc(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]:
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    true_positives = torch.cumsum(sorted_targets, dim=0).float()
    false_positives = torch.cumsum(1 - sorted_targets, dim=0).float()
    distinct = torch.ones_like(sorted_scores, dtype=torch.bool)
    distinct[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    tpr = true_positives[distinct] / max(float(targets.sum().item()), 1.0)
    fpr = false_positives[distinct] / max(float((targets == 0).sum().item()), 1.0)
    thresholds = sorted_scores[distinct]
    tpr = torch.cat([torch.tensor([0.0]), tpr.cpu()])
    fpr = torch.cat([torch.tensor([0.0]), fpr.cpu()])
    thresholds = torch.cat([torch.tensor([float("inf")]), thresholds.cpu()])
    auc = float(torch.trapz(tpr, fpr).item())

    accuracies = (tpr * targets.sum() + (1.0 - fpr) * (targets == 0).sum()) / targets.numel()
    best_idx = int(torch.argmax(accuracies).item())
    best_threshold = float(thresholds[best_idx].item())
    best_accuracy = float(accuracies[best_idx].item())
    eer_idx = int(torch.argmin(torch.abs(fpr - (1.0 - tpr))).item())
    eer = float(((fpr[eer_idx] + (1.0 - tpr[eer_idx])) / 2.0).item())
    return fpr, tpr, thresholds, auc, best_threshold, best_accuracy, eer


def compute_verification_metrics(scores: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict[str, float | int]:
    predictions = scores >= threshold
    positives = targets == 1
    negatives = targets == 0
    true_positive = int((predictions & positives).sum().item())
    false_positive = int((predictions & negatives).sum().item())
    true_negative = int((~predictions & negatives).sum().item())
    false_negative = int((~predictions & positives).sum().item())

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    return {
        "verification_accuracy": divide(true_positive + true_negative, targets.numel()),
        "verification_precision": precision,
        "verification_recall": recall,
        "verification_f1": divide(2.0 * precision * recall, precision + recall),
        "verification_specificity": divide(true_negative, true_negative + false_positive),
        "verification_true_positive": true_positive,
        "verification_false_positive": false_positive,
        "verification_true_negative": true_negative,
        "verification_false_negative": false_negative,
    }


def save_verification_confusion_matrix(output_dir: Path, metrics: dict[str, float | int]) -> None:
    matrix = [
        [int(metrics["verification_true_negative"]), int(metrics["verification_false_positive"])],
        [int(metrics["verification_false_negative"]), int(metrics["verification_true_positive"])],
    ]
    with (output_dir / "test_verification_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", "different_identity", "same_identity"])
        writer.writerow(["different_identity", *matrix[0]])
        writer.writerow(["same_identity", *matrix[1]])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Different", "Same"])
    ax.set_yticks([0, 1], labels=["Different", "Same"])
    ax.set_xlabel("Predicted identity relationship")
    ax.set_ylabel("Actual identity relationship")
    ax.set_title("Test Verification Confusion Matrix")
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "test_verification_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_verification_pairs(
    output_dir: Path,
    pairs: Sequence[tuple[int, int, int]],
    scores: torch.Tensor,
    labels: torch.Tensor,
    classes: Sequence[str],
    paths: Sequence[str],
) -> None:
    with (output_dir / "test_verification_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["left_index", "right_index", "left_class", "right_class", "is_same", "similarity", "left_path", "right_path"]
        )
        for (left_idx, right_idx, is_same), score in zip(pairs, scores.tolist()):
            writer.writerow(
                [
                    left_idx,
                    right_idx,
                    classes[int(labels[left_idx])],
                    classes[int(labels[right_idx])],
                    is_same,
                    score,
                    paths[left_idx],
                    paths[right_idx],
                ]
            )


def save_roc(output_dir: Path, roc: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float, float]) -> None:
    fpr, tpr, thresholds, auc, _, _, _ = roc
    with (output_dir / "test_roc_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "fpr", "tpr"])
        writer.writerows(zip(thresholds.tolist(), fpr.tolist(), tpr.tolist()))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plot generation")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr.numpy(), tpr.numpy(), label=f"Triplet Loss ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Test Face Verification ROC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "test_roc_curve.png", dpi=160)
    plt.close(fig)


def save_similarity_distribution(output_dir: Path, scores: torch.Tensor, targets: torch.Tensor, threshold: float) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[targets == 1].numpy(), bins=50, alpha=0.65, label="Same identity")
    ax.hist(scores[targets == 0].numpy(), bins=50, alpha=0.65, label="Different identities")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Best threshold = {threshold:.4f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Test Verification Similarity Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "test_similarity_distribution.png", dpi=160)
    plt.close(fig)


def save_summary(output_dir: Path, results: dict[str, float | int | str]) -> None:
    with (output_dir / "test_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with (output_dir / "test_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(results.items())


def plot_training_history(history_path: Path, output_dir: Path) -> None:
    if not history_path.is_file():
        return
    history = load_json(history_path)
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for split in ("train", "val"):
        axes[0, 0].plot(epochs, [row[f"{split}_loss"] for row in history], label=f"{split}_loss")
        axes[0, 1].plot(epochs, [row[f"{split}_triplet_accuracy"] for row in history], label=f"{split}_triplet_accuracy")
        axes[1, 0].plot(epochs, [row[f"{split}_positive_distance"] for row in history], label=f"{split}_positive_distance")
        axes[1, 0].plot(epochs, [row[f"{split}_negative_distance"] for row in history], label=f"{split}_negative_distance")
    axes[1, 1].plot(epochs, [row["lr"] for row in history], label="lr")
    for ax, title in zip(axes.flat, ("Triplet Loss", "Triplet Accuracy", "Embedding Distances", "Learning Rate")):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve() if args.metadata else checkpoint_path.parent / "training_metadata.json"
    history_path = Path(args.history).expanduser().resolve() if args.history else checkpoint_path.parent / "training_history.json"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint_path.parent.with_name(f"{checkpoint_path.parent.name}_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(metadata_path)
    config = payload["config"]
    data_root = args.data_root or config["data_root"]
    batch_size = args.batch_size or int(config.get("val_batch_size", config["batch_size"]))
    num_workers = int(config.get("num_workers", 4)) if args.num_workers < 0 else args.num_workers
    seed = int(config.get("seed", 42)) if args.seed < 0 else args.seed
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    dataset = build_dataset(
        data_root=data_root,
        split="test",
        image_size=int(config["image_size"]),
        grayscale=True,
        resize_size=int(config["resize_size"]) if int(config.get("resize_size", 0)) > 0 else None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    triplet_loader = DataLoader(
        TripletFaceDataset(dataset, train=False, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = NormalizedEmbeddingModel(
        resnet18_face(
            input_size=int(config["image_size"]),
            embedding_size=int(config["embedding_size"]),
            input_channels=1,
            use_se=bool(config.get("use_se", False)),
            dropout=float(config.get("dropout", 0.35)),
        )
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(checkpoint.get("model_state", checkpoint))
    model.backbone.load_state_dict(state_dict)

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"test images: {len(dataset)} | eligible test triplets: {len(triplet_loader.dataset)}")
    triplet_results = evaluate_triplets(model, triplet_loader, device, float(config["margin"]), args.no_progress)
    embeddings, labels = extract_embeddings(model, loader, device, args.no_progress)
    pairs = build_verification_pairs(labels, args.negative_ratio, args.max_positive_pairs, args.max_negative_pairs, seed)
    scores = score_pairs(embeddings, pairs)
    targets = torch.tensor([is_same for _, _, is_same in pairs], dtype=torch.long)
    roc = compute_roc(scores, targets)
    _, _, _, auc, best_threshold, best_accuracy, eer = roc
    verification_metrics = compute_verification_metrics(scores, targets, best_threshold)

    results: dict[str, float | int | str] = {
        **triplet_results,
        "verification_auc": auc,
        "verification_best_threshold": best_threshold,
        "verification_best_accuracy": best_accuracy,
        **verification_metrics,
        "verification_eer": eer,
        "verification_positive_pairs": int(targets.sum().item()),
        "verification_negative_pairs": int((targets == 0).sum().item()),
        "verification_pairs": int(targets.numel()),
        "checkpoint": str(checkpoint_path),
    }
    save_summary(output_dir, results)
    save_roc(output_dir, roc)
    save_similarity_distribution(output_dir, scores, targets, best_threshold)
    save_verification_confusion_matrix(output_dir, verification_metrics)
    save_verification_pairs(output_dir, pairs, scores, labels, dataset.classes, [path for path, _ in dataset.samples])
    plot_training_history(history_path, output_dir)

    print(
        f"test triplet loss {triplet_results['triplet_loss']:.4f} "
        f"acc {triplet_results['triplet_accuracy']:.4f} "
        f"pos_dist {triplet_results['positive_distance']:.4f} "
        f"neg_dist {triplet_results['negative_distance']:.4f}"
    )
    print(
        f"verification auc {auc:.4f} accuracy {best_accuracy:.4f} "
        f"eer {eer:.4f} threshold {best_threshold:.4f}"
    )
    print(f"saved results to {output_dir}")


if __name__ == "__main__":
    main()
