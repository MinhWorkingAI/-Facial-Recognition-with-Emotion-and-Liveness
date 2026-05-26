"""
Move face crops from screenshots into training dataset folders.

Each capture from the API is a pair of files in SCREENSHOTS_DIR:
  <timestamp>_frame.jpg          — full camera frame
  <timestamp>_predictions.json   — pipeline output (faces, emotion, liveness, etc.)

This script crops each face, copies it into Train or Test folders under the right
dataset (AffectNet, LCC_FASD, …), then deletes every screenshot pair (merged or not)
and any orphan *_frame.jpg / *_predictions.json files left behind.

Edit SETTINGS below, then run:
  python backend/captures/merge_captures.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image

# backend/captures/ and repo root (parent of backend/)
CAPTURES_PKG = Path(__file__).resolve().parent
REPO_ROOT = CAPTURES_PKG.parents[1]

# ---------------------------------------------------------------------------
# SETTINGS — Changes based on base repo and where the training files are stored
# ---------------------------------------------------------------------------

SCREENSHOTS_DIR = "backend/captures/screenshots"

# Minimum face-detector score (0.0–1.0) to crop a face at all
MIN_DETECTION_CONFIDENCE = 0.7

# Minimum emotion / liveness score (0.0–1.0) to copy a crop into a dataset
MIN_LABEL_CONFIDENCE = 0.7

# Fraction of crops that go to training (rest go to testing). 0.7 = 70% train
TRAIN_SPLIT_RATIO = 0.7

# Datasets to fill. Each entry = one model’s data (emotion, anti-spoofing, …).
# Crops land in: <train_dir or test_dir>/<label>/capture_<stamp>_faceN.jpg
TARGETS = [
    {
        "name": "anti_spoofing",
        "train_dir": "LCC_FASD/LCC_FASD_training",
        "test_dir": "LCC_FASD/LCC_FASD_evaluation",
        "label_from": "liveness",
    },
    {
        "name": "emotion_affectnet",
        "train_dir": "AffectNet/Train",
        "test_dir": "AffectNet/Test",
        "label_from": "emotion", 
    },
]

# Future face recognition — uncomment and add to TARGETS when ready:
# PREDICTION_KEYS["recognition"] = ("recognition", "label", "confidence")
# TARGETS.append({
#     "name": "face_recognition",
#     "train_dir": "path/to/train_data",
#     "test_dir": "path/to/test_data",
#     "label_from": "recognition",
# })

# Maps label_from → (JSON section, label field, confidence field)
PREDICTION_KEYS = {
    "emotion": ("emotion", "label", "confidence"),
    "liveness": ("anti_spoofing", "label", "confidence"),
}


def to_repo_path(path_text: str) -> Path:
    """Resolve a repo-relative path like AffectNet/Train to an absolute Path."""
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def crop_face(image: Image.Image, face: dict) -> Image.Image:
    """Cut the face out of the full frame using the normalised bbox (0–1) in JSON."""
    box = face["face"]["bbox"]
    width, height = image.size
    left = int(box["x"] * width)
    top = int(box["y"] * height)
    right = int((box["x"] + box["w"]) * width)
    bottom = int((box["y"] + box["h"]) * height)
    return image.crop((left, top, right, bottom))


def label_for_dataset(face: dict, target: dict) -> tuple[str, float] | None:
    """
    Read the folder name and confidence for one target (e.g. emotion → "happy", 0.92).
    Returns None if there is no label to use.
    """
    kind = target["label_from"]
    section_name, label_key, score_key = PREDICTION_KEYS[kind]
    block = face.get(section_name) or {}

    text = str(block.get(label_key, "")).strip()
    try:
        score = float(block.get(score_key, 0.0))
    except (TypeError, ValueError):
        score = 0.0

    if not text:
        return None
    return text, score


def is_train_split() -> bool:
    """Randomly pick train vs test; True approximately TRAIN_SPLIT_RATIO of the time."""
    return random.random() < TRAIN_SPLIT_RATIO


def delete_pair(screenshots_dir: Path, stamp: str) -> None:
    (screenshots_dir / f"{stamp}_frame.jpg").unlink(missing_ok=True)
    (screenshots_dir / f"{stamp}_predictions.json").unlink(missing_ok=True)


def cleanup_leftovers(screenshots_dir: Path) -> int:
    """Remove orphan halves and any capture files still in screenshots."""
    removed = 0
    for json_path in list(screenshots_dir.glob("*_predictions.json")):
        stamp = json_path.stem.removesuffix("_predictions")
        if not (screenshots_dir / f"{stamp}_frame.jpg").is_file():
            json_path.unlink(missing_ok=True)
            removed += 1
    for jpg_path in list(screenshots_dir.glob("*_frame.jpg")):
        stamp = jpg_path.stem.removesuffix("_frame")
        if not (screenshots_dir / f"{stamp}_predictions.json").is_file():
            jpg_path.unlink(missing_ok=True)
            removed += 1
    return removed


def main() -> int:
    screenshots_dir = to_repo_path(SCREENSHOTS_DIR)
    if not screenshots_dir.is_dir():
        print(f"Screenshots folder not found: {screenshots_dir}", file=sys.stderr)
        return 1
    if not TARGETS:
        print("No targets configured in TARGETS.", file=sys.stderr)
        return 1

    total_crops = 0
    merged_pairs = 0

    for json_path in sorted(screenshots_dir.glob("*_predictions.json")):
        stamp = json_path.stem.removesuffix("_predictions")
        jpg_path = screenshots_dir / f"{stamp}_frame.jpg"
        if not jpg_path.is_file():
            json_path.unlink(missing_ok=True)
            continue

        predictions = json.loads(json_path.read_text(encoding="utf-8"))
        faces = predictions.get("faces") or []
        if not faces:
            delete_pair(screenshots_dir, stamp)
            continue

        frame = Image.open(jpg_path).convert("RGB")
        crops_saved_for_this_pair = 0  # resets per frame; used to decide deletion

        # --- Each face in the frame ---
        for face_index, face in enumerate(faces):
            detection_score = float((face.get("face") or {}).get("detection_confidence", 0.0))
            if detection_score < MIN_DETECTION_CONFIDENCE:
                continue  # detector not confident enough

            try:
                crop = crop_face(frame, face)
            except (KeyError, TypeError, ValueError):
                continue  # bad or missing bbox

            # --- Copy the same crop into each dataset (emotion, liveness, …) ---
            for target in TARGETS:
                result = label_for_dataset(face, target)
                if result is None:
                    continue

                folder_name, label_score = result
                if label_score < MIN_LABEL_CONFIDENCE:
                    continue  # model not confident in happy/real/etc.

                # 70/30 (or TRAIN_SPLIT_RATIO) → train_dir vs test_dir
                use_train = is_train_split()
                dataset_dir = target["train_dir"] if use_train else target["test_dir"]
                split = "train" if use_train else "test"
                out_file = (
                    to_repo_path(dataset_dir)
                    / folder_name
                    / f"capture_{stamp}_face{face_index}.jpg"
                )
                out_file.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out_file, format="JPEG", quality=95)
                print(
                    f"{target['name']}:"
                    f"{jpg_path.name} "
                    f"face{face_index} "
                    "to:"
                    f"{split}/{folder_name} ({label_score:.2f})"
                    f"{out_file}"
                )
                crops_saved_for_this_pair += 1
                total_crops += 1

        if crops_saved_for_this_pair:
            merged_pairs += 1
        delete_pair(screenshots_dir, stamp)

    leftovers_removed = cleanup_leftovers(screenshots_dir)
    print(
        f"Done. {total_crops} crop(s) from {merged_pairs} pair(s). "
        f"Removed {leftovers_removed} leftover file(s) from screenshots."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
