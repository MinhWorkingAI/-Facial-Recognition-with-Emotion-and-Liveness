"""
download_model.py
=================
Handles downloading the InsightFace buffalo_l model pack, verifying its
SHA-256 integrity, and unpacking the relevant ONNX model files into /model.

This module is imported automatically by face_detection.py when no model is
found in the /model directory, so end-users rarely need to call it directly.

Public API
----------
    ensure_model(model_dir: str | Path) -> Path
        Top-level entry point. Downloads + verifies + unpacks if needed.
        Returns the path to the unpacked model directory.

    download_buffalo_l(dest_dir: str | Path) -> Path
        Downloads the zip to dest_dir, verifies SHA-256, returns zip path.

    unpack_buffalo_l(zip_path: str | Path, model_dir: str | Path) -> Path
        Unpacks the zip into model_dir/buffalo_l/, returns that directory.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Model source — single authoritative URL for the buffalo_l pack
# ---------------------------------------------------------------------------
BUFFALO_L_URL: str = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)

# SHA-256 of the official buffalo_l.zip (v0.7 release).
# Recompute with: sha256sum buffalo_l.zip
BUFFALO_L_SHA256: str = (
    "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"  
)

# Name of the subfolder created inside model_dir after unpacking.
UNPACKED_DIR_NAME: str = "buffalo_l"

# The specific ONNX file we need for face detection inside the pack.
DETECTOR_ONNX_NAME: str = "det_10g.onnx"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the SHA-256 digest of a file in streaming chunks.

    Parameters
    ----------
    path : Path
        File to hash.
    chunk_size : int
        Read chunk size in bytes (default 1 MiB).

    Returns
    -------
    str
        Lowercase hex digest string (64 characters).
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """
    urllib.request.urlretrieve progress callback.

    Parameters
    ----------
    block_num : int
        Number of blocks transferred so far.
    block_size : int
        Size of each block in bytes.
    total_size : int
        Total file size reported by the server (-1 if unknown).
    """
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  ({downloaded // 1_048_576} / {total_size // 1_048_576} MiB)", end="", flush=True)
    else:
        print(f"\r  Downloaded {downloaded // 1_048_576} MiB …", end="", flush=True)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def download_buffalo_l(dest_dir: str | Path) -> Path:
    """
    Download buffalo_l.zip into *dest_dir* and verify its SHA-256 digest.

    If the zip already exists in *dest_dir* and passes the integrity check it
    is reused without re-downloading.

    Parameters
    ----------
    dest_dir : str | Path
        Directory where the zip will be saved (created if absent).

    Returns
    -------
    Path
        Absolute path to the verified buffalo_l.zip file.

    Raises
    ------
    ValueError
        If the downloaded file's SHA-256 does not match BUFFALO_L_SHA256.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "buffalo_l.zip"

    if zip_path.exists():
        print(f"[download_model] Found existing zip at {zip_path}. Verifying …")
        digest = _sha256_of_file(zip_path)
        if digest == BUFFALO_L_SHA256:
            print("[download_model] Integrity OK — skipping download.")
            return zip_path
        print("[download_model] Integrity check FAILED — re-downloading.")
        zip_path.unlink()

    print(f"[download_model] Downloading buffalo_l.zip from:\n  {BUFFALO_L_URL}")
    urllib.request.urlretrieve(BUFFALO_L_URL, zip_path, reporthook=_progress_hook)
    print()  # newline after progress bar

    print("[download_model] Verifying SHA-256 …")
    digest = _sha256_of_file(zip_path)
    if digest != BUFFALO_L_SHA256:
        zip_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch!\n"
            f"  Expected : {BUFFALO_L_SHA256}\n"
            f"  Got      : {digest}\n"
            "The downloaded file may be corrupted or tampered with."
        )

    print(f"[download_model] Integrity OK  ({digest[:16]}…)")
    return zip_path


def unpack_buffalo_l(zip_path: str | Path, model_dir: str | Path) -> Path:
    """
    Unpack buffalo_l.zip into *model_dir*/buffalo_l/.

    Already-extracted files are not overwritten; the function is idempotent.

    Parameters
    ----------
    zip_path : str | Path
        Path to the verified buffalo_l.zip.
    model_dir : str | Path
        Root model directory (typically Face_detection_module/model).

    Returns
    -------
    Path
        Path to the unpacked buffalo_l/ directory inside model_dir.
    """
    zip_path = Path(zip_path)
    model_dir = Path(model_dir)
    unpack_dir = model_dir / UNPACKED_DIR_NAME

    if unpack_dir.exists() and any(unpack_dir.iterdir()):
        print(f"[download_model] Models already unpacked at {unpack_dir} — skipping.")
        return unpack_dir

    unpack_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download_model] Unpacking {zip_path.name} → {unpack_dir} …")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            # Strip any leading directory component that is exactly the zip's
            # stem (some archives wrap everything in a top-level folder).
            parts = Path(member).parts
            if len(parts) > 1 and parts[0].lower() == UNPACKED_DIR_NAME.lower():
                out_relative = Path(*parts[1:])
            else:
                out_relative = Path(member)

            out_path = unpack_dir / out_relative
            if member.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

            pct = i / total * 100
            print(f"\r  Extracting … {pct:5.1f}%  [{i}/{total}]", end="", flush=True)

    print(f"\n[download_model] Unpacked {total} file(s) to {unpack_dir}")
    return unpack_dir


def ensure_model(model_dir: str | Path) -> Path:
    """
    Top-level entry point: guarantee that the buffalo_l ONNX models exist.

    Steps
    -----
    1. Check whether model_dir/buffalo_l/det_10g.onnx already exists.
    2. If not, download buffalo_l.zip into model_dir, verify SHA-256.
    3. Unpack into model_dir/buffalo_l/.
    4. Return the unpacked directory path.

    Parameters
    ----------
    model_dir : str | Path
        Root model directory (Face_detection_module/model by convention).

    Returns
    -------
    Path
        Path to the unpacked model directory (model_dir/buffalo_l/).
    """
    model_dir = Path(model_dir)
    unpack_dir = model_dir / UNPACKED_DIR_NAME
    detector_path = unpack_dir / DETECTOR_ONNX_NAME

    if detector_path.exists():
        print(f"[download_model] Detector model found: {detector_path}")
        return unpack_dir

    print("[download_model] Model not found — initiating download …")
    zip_path = download_buffalo_l(model_dir)
    unpack_buffalo_l(zip_path, model_dir)

    if not detector_path.exists():
        raise FileNotFoundError(
            f"Expected {DETECTOR_ONNX_NAME} inside {unpack_dir} after unpacking, "
            "but it was not found. The zip structure may have changed."
        )

    print(f"[download_model] Setup complete. Detector: {detector_path}")
    return unpack_dir


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "model"
    ensure_model(target)
