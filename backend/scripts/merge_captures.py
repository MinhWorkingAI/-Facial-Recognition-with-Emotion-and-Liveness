"""
Copy pipeline face crops from backend/captures into training dataset folders.

  python backend/scripts/merge_captures.py --dry-run
  python backend/scripts/merge_captures.py --reprocess
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "merge_config.yaml"
DEFAULT_MANIFEST = SCRIPT_DIR / ".merge_manifest.json"

# label_from in merge_config.yaml -> (JSON section, label field, confidence field)
JSON_FIELDS = {
    "emotion": ("emotion", "label", "confidence"),
    "liveness": ("anti_spoofing", "label", "confidence"),
    "recognition": ("recognition", "label", "confidence"),
}

# Built-in targets: split aliases, optional root auto-discovery
TARGET_SPECS: dict[str, dict[str, Any]] = {
    "anti_spoofing": {
        "splits": {
            "train": "LCC_FASD_training",
            "training": "LCC_FASD_training",
            "development": "LCC_FASD_development",
            "dev": "LCC_FASD_development",
            "evaluation": "LCC_FASD_evaluation",
            "test": "LCC_FASD_evaluation",
        },
        "root_markers": ("LCC_FASD_training", "LCC_FASD_development", "LCC_FASD_evaluation"),
        "nest_under": "LCC_FASD",
    },
    "emotion_affectnet": {
        "splits": {"train": "Train", "test": "Test"},
        "root_markers": ("Train", "Test"),
        "nest_under": "AffectNet",
        "extra_roots": ("training_module/emotion_module/affectnet/AffectNet",),
    },
    "face_recognition": {
        "splits": {"train": "train_data", "val": "val_data", "test": "test_data"},
        "preserve_label_case": True,
    },
}


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_repo_root(config: dict[str, Any]) -> Path:
    raw = (config.get("repo_root") or "").strip()
    if raw in ("", "."):
        return REPO_ROOT
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_path(repo_root: Path, raw: str, field_name: str) -> Path:
    value = (raw or "").strip()
    if not value:
        raise ValueError(f"Missing required path: {field_name}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def resolve_dataset_root(
    configured: Path,
    repo_root: Path,
    *,
    markers: tuple[str, ...] = (),
    nest_under: str | None = None,
    extra_roots: tuple[str, ...] = (),
) -> Path:
    """Find the folder that contains expected split subdirectories."""
    if not markers:
        return configured

    candidates: list[Path] = [configured]
    if nest_under:
        candidates.append(configured / nest_under)
        candidates.append(repo_root / nest_under)
    for rel in extra_roots:
        candidates.append(repo_root / rel)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if any((resolved / name).is_dir() for name in markers):
            return resolved
    return configured


def min_confidence(config: dict[str, Any], target: dict[str, Any]) -> float:
    if target.get("min_confidence") is not None:
        return float(target["min_confidence"])
    return float(config.get("min_confidence", 0.0))


def load_manifest(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(x) for x in data.get("ingested", [])} if isinstance(data, dict) else set()


def save_manifest(path: Path, ingested: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ingested": sorted(ingested)}, indent=2), encoding="utf-8")


def sanitize_folder(label: str, *, preserve_case: bool = False) -> str:
    if preserve_case:
        cleaned = label.strip()
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", cleaned).strip(" .")
    else:
        cleaned = re.sub(r"[^\w\-.]+", "_", label.strip().lower()).strip("._")
    if not cleaned:
        raise ValueError("empty label after sanitization")
    return cleaned


def crop_face(image: Image.Image, face: dict[str, Any]) -> Image.Image:
    box = face["face"]["bbox"]
    w, h = image.size
    left = max(0, min(int(box["x"] * w), w - 1))
    top = max(0, min(int(box["y"] * h), h - 1))
    right = max(left + 1, min(int((box["x"] + box["w"]) * w), w))
    bottom = max(top + 1, min(int((box["y"] + box["h"]) * h), h))
    return image.crop((left, top, right, bottom))


def read_prediction(target_name: str, target: dict[str, Any], face: dict[str, Any]) -> Prediction | None:
    spec = TARGET_SPECS.get(target_name)
    if spec is None:
        raise ValueError(f"Unknown target {target_name!r}")

    label_from = str(target.get("label_from", "emotion")).strip().lower()
    section_key, label_key, conf_key = JSON_FIELDS[label_from]
    section = face.get(section_key) or {}
    raw_label = section.get(label_key, "")
    try:
        confidence = float(section.get(conf_key, 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    if label_from == "recognition":
        if target.get("require_live") and str((face.get("anti_spoofing") or {}).get("label", "")).lower() != "real":
            return None
        if target.get("only_matched") and not (face.get("recognition") or {}).get("matched"):
            return None
        if not raw_label or str(raw_label).lower() == "unknown":
            return None
        return Prediction(
            label=sanitize_folder(str(raw_label), preserve_case=bool(spec.get("preserve_label_case"))),
            confidence=confidence,
        )

    if not raw_label:
        return None
    return Prediction(label=sanitize_folder(str(raw_label)), confidence=confidence)


def split_folder(target_name: str, target: dict[str, Any]) -> str:
    spec = TARGET_SPECS[target_name]
    split = str(target.get("split", "train")).strip().lower()
    folder = spec["splits"].get(split)
    if not folder:
        raise ValueError(
            f"targets.{target_name}.split={split!r} invalid; "
            f"use one of {sorted(spec['splits'])}"
        )
    return folder


def output_dir(target_name: str, target: dict[str, Any], repo_root: Path, class_label: str) -> Path:
    spec = TARGET_SPECS[target_name]
    root = resolve_path(repo_root, str(target["root"]), f"targets.{target_name}.root")
    root = resolve_dataset_root(
        root,
        repo_root,
        markers=tuple(spec.get("root_markers", ())),
        nest_under=spec.get("nest_under"),
        extra_roots=tuple(spec.get("extra_roots", ())),
    )
    return root / split_folder(target_name, target) / class_label


def enabled_targets(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    targets = config.get("targets") or {}
    if not isinstance(targets, dict):
        raise ValueError("config.targets must be a mapping")
    return [
        (name, cfg)
        for name, cfg in targets.items()
        if isinstance(cfg, dict) and cfg.get("enabled") and name in TARGET_SPECS
    ]


def list_capture_stamps(captures_dir: Path) -> list[str]:
    stamps = []
    for json_path in sorted(captures_dir.glob("*_predictions.json")):
        stamp = json_path.name.removesuffix("_predictions.json")
        if (captures_dir / f"{stamp}_frame.jpg").is_file():
            stamps.append(stamp)
    return stamps


def merge_capture(
    stamp: str,
    captures_dir: Path,
    targets: list[tuple[str, dict[str, Any]]],
    repo_root: Path,
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> tuple[int, list[str]]:
    json_path = captures_dir / f"{stamp}_predictions.json"
    frame_path = captures_dir / f"{stamp}_frame.jpg"

    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    faces = payload.get("faces", [])
    if not faces:
        return 0, ["no faces in JSON"]

    min_detection = float(config.get("min_detection_confidence", 0.0))
    image = Image.open(frame_path).convert("RGB")
    written = 0
    notes: list[str] = []

    for face_index, face in enumerate(faces):
        det_conf = float(face.get("face", {}).get("detection_confidence", 0.0))
        if det_conf < min_detection:
            notes.append(f"face {face_index}: detection {det_conf:.3f} < {min_detection:.3f}")
            continue

        try:
            crop = crop_face(image, face)
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"face {face_index}: bad bbox ({exc})")
            continue

        for target_name, target in targets:
            threshold = min_confidence(config, target)
            try:
                prediction = read_prediction(target_name, target, face)
            except ValueError as exc:
                notes.append(f"face {face_index} / {target_name}: {exc}")
                continue

            if prediction is None:
                continue
            if prediction.confidence < threshold:
                notes.append(
                    f"face {face_index} / {target_name}: "
                    f"{prediction.label} {prediction.confidence:.3f} < {threshold:.3f}"
                )
                continue

            out_path = output_dir(target_name, target, repo_root, prediction.label) / (
                f"capture_{stamp}_face{face_index}.jpg"
            )
            if dry_run:
                print(f"[dry-run] {prediction.label} ({prediction.confidence:.3f}) -> {out_path}")
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out_path, format="JPEG", quality=95)
            written += 1

    return written, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge pipeline captures into training datasets.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reprocess", action="store_true", help="Ignore manifest.")
    args = parser.parse_args(argv)

    config = load_config(args.config.resolve())
    repo_root = resolve_repo_root(config)
    captures_dir = resolve_path(repo_root, config.get("captures_dir", ""), "captures_dir")

    manifest_raw = (config.get("manifest_path") or "").strip()
    manifest_path = (
        resolve_path(repo_root, manifest_raw, "manifest_path") if manifest_raw else DEFAULT_MANIFEST
    )

    targets = enabled_targets(config)
    if not targets:
        print("No enabled targets in merge_config.yaml.", file=sys.stderr)
        return 1
    if not captures_dir.is_dir():
        print(f"Captures directory not found: {captures_dir}", file=sys.stderr)
        return 1

    ingested = load_manifest(manifest_path)
    stamps = list_capture_stamps(captures_dir)
    pending = stamps if args.reprocess else [s for s in stamps if s not in ingested]

    print(f"Repo:      {repo_root}")
    print(f"Captures:  {captures_dir} ({len(stamps)} pairs, {len(pending)} to process)")
    print(f"Targets:   {', '.join(n for n, _ in targets)}")
    if args.dry_run:
        print("Mode:      dry-run")
    if args.reprocess:
        print("Mode:      reprocess")

    total_written = 0
    merged: list[str] = []

    for stamp in pending:
        try:
            count, notes = merge_capture(
                stamp, captures_dir, targets, repo_root, config, dry_run=args.dry_run
            )
        except Exception as exc:
            print(f"[skip] {stamp}: {exc}", file=sys.stderr)
            continue

        for note in notes:
            print(f"  {stamp}: {note}")

        if count > 0:
            total_written += count
            merged.append(stamp)
            if not args.dry_run:
                ingested.add(stamp)
        else:
            print(f"[skip] {stamp}: nothing written")

    if not args.dry_run and merged:
        save_manifest(manifest_path, ingested)

    print(f"Done. {total_written} image(s) from {len(merged)} capture(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
