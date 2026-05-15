"""
face_detection_demo.py
======================
Interactive demo for the Face_detection_module.

Captures frames from a webcam (or a video file), runs face detection via
:class:`face_detection.FaceDetector`, and renders the results in a live
OpenCV window showing:

    * Bounding boxes  (green rectangles)
    * Confidence scores  (text above each box)
    * 5 facial landmarks  (coloured circles: eyes=blue, nose=yellow, mouth=red)
    * Real-time FPS counter  (top-left corner)

This file imports from ``face_detection.py`` and must NOT contain any
detection or model-loading logic of its own.

Usage
-----
::

    # Webcam (device 0)
    python face_detection_demo.py

    # Specific camera index
    python face_detection_demo.py --camera 1

    # Video file
    python face_detection_demo.py --video path/to/clip.mp4

    # Custom thresholds
    python face_detection_demo.py --conf 0.6 --nms 0.3

Controls
--------
    Q  or  ESC   — quit
    S            — save current frame to  demo_snapshot_<timestamp>.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Import the core detector from this package
# ---------------------------------------------------------------------------
# Allow running the demo both as a module and as a standalone script.
try:
    from .face_detection import FaceDetector, DEFAULT_CONF_THRESHOLD, DEFAULT_NMS_THRESHOLD
except ImportError:
    # Standalone: add parent to path so the bare import works.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from Face_detection_module.face_detection import (
        FaceDetector,
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_NMS_THRESHOLD,
    )

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------

# Bounding box
_BBOX_COLOR: Tuple[int, int, int] = (0, 255, 0)       # Green (BGR)
_BBOX_THICKNESS: int = 2

# Confidence label
_LABEL_COLOR: Tuple[int, int, int] = (0, 255, 0)
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE: float = 0.55
_LABEL_THICKNESS: int = 1

# Landmark colours (BGR) per keypoint index:
#   0=left eye, 1=right eye  → blue
#   2=nose                   → yellow
#   3=left mouth, 4=right mouth → red
_LM_COLORS: list[Tuple[int, int, int]] = [
    (255, 100, 0),   # left eye   — blue-ish
    (255, 100, 0),   # right eye  — blue-ish
    (0, 220, 220),   # nose       — yellow-ish
    (0, 80, 255),    # left mouth — red-ish
    (0, 80, 255),    # right mouth — red-ish
]
_LM_RADIUS: int = 3
_LM_THICKNESS: int = -1  # filled circle

# FPS counter
_FPS_COLOR: Tuple[int, int, int] = (0, 255, 255)      # Cyan (BGR)
_FPS_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FPS_SCALE: float = 0.7
_FPS_THICKNESS: int = 2

# FPS smoothing window (number of frames)
_FPS_SMOOTH: int = 30

# Anti-spoofing image size (must match MobileNetV2 training input)
IMG_SIZE: Tuple[int, int] = (224, 224)

# Default anti-spoofing model path (best checkpoint from training)
# Points to SavedModel directory — resave from notebook with:
#   model.save("model/best_fasd_mobilenetv2_savedmodel")
_DEFAULT_SPOOF_MODEL_PATH: Path = (
    Path(__file__).parent.parent
    / "anti_spoofing_module_MobileNetV2"
    / "model"
    / "best_fasd_mobilenetv2_savedmodel"
)

# Class names as stored by image_dataset_from_directory (alphabetical)
# LCC_FASD dataset folders: real/, spoof/
_SPOOF_CLASS_NAMES: list[str] = ["real", "spoof"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_detections(
    frame_bgr: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    landmarks: np.ndarray,
) -> None:
    """
    Render bounding boxes, confidence labels, and facial landmarks onto
    *frame_bgr* **in-place**.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Shape ``(H, W, 3)``, dtype uint8, BGR colour order (OpenCV native).
        Modified in-place.
    bboxes : np.ndarray
        Shape ``(N, 4)`` float32, ``[x1, y1, x2, y2]``.
    scores : np.ndarray
        Shape ``(N,)`` float32, confidence in ``[0, 1]``.
    landmarks : np.ndarray
        Shape ``(N, 5, 2)`` float32, pixel xy per keypoint.
    """
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i].astype(int)
        conf = float(scores[i])

        # -- Bounding box
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), _BBOX_COLOR, _BBOX_THICKNESS)

        # -- Confidence label (above box, clamped so it stays on screen)
        label = f"{conf:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS)
        label_y = max(y1 - 6, lh + 4)
        cv2.putText(
            frame_bgr, label,
            (x1, label_y),
            _LABEL_FONT, _LABEL_SCALE, _LABEL_COLOR, _LABEL_THICKNESS,
            cv2.LINE_AA,
        )

        # -- Landmarks
        for k in range(5):
            lx, ly = int(landmarks[i, k, 0]), int(landmarks[i, k, 1])
            cv2.circle(frame_bgr, (lx, ly), _LM_RADIUS, _LM_COLORS[k], _LM_THICKNESS)


def draw_fps(frame_bgr: np.ndarray, fps: float) -> None:
    """
    Render the FPS counter in the top-left corner of *frame_bgr* in-place.

    Parameters
    ----------
    frame_bgr : np.ndarray
        OpenCV frame (modified in-place).
    fps : float
        Frames-per-second value to display.
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame_bgr, text,
        (10, 32),
        _FPS_FONT, _FPS_SCALE, _FPS_COLOR, _FPS_THICKNESS,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Anti-spoofing helper
# ---------------------------------------------------------------------------

def predict_real_or_spoof(
    face_rgb: np.ndarray,
    spoof_model: "tf.keras.Model",
    class_names: list[str],
) -> tuple[str, float]:
    """
    Run the anti-spoofing model on a single cropped face.

    Preprocessing mirrors the notebook pipeline:
      - resize to IMG_SIZE (224×224)
      - cast to float32  (raw 0-255 pixel values)
      - preprocess_input (mobilenet_v2 scaling to [-1, 1]) is **baked into
        the saved model**, so no manual scaling is needed here.

    Parameters
    ----------
    face_rgb : np.ndarray
        Cropped face region in RGB order, uint8, shape (H, W, 3).
    spoof_model : tf.keras.Model
        Loaded anti-spoofing Keras model.
    class_names : list[str]
        Ordered class labels matching the model's output indices
        (e.g. ["real", "spoof"]).

    Returns
    -------
    label : str
        Predicted class name.
    confidence : float
        Confidence score in [0, 1].
    """
    # -- Preprocess: resize and add batch dim
    img = cv2.resize(face_rgb, IMG_SIZE)          # (224, 224, 3)
    x = img.astype(np.float32)                    # keep raw 0-255; model scales internally
    x = np.expand_dims(x, axis=0)                 # (1, 224, 224, 3)

    # -- Predict
    pred = spoof_model.predict(x, verbose=0)

    # -- Postprocess: works for both binary sigmoid and multi-class softmax
    if pred.shape[-1] == 1:
        # binary output
        score = float(pred[0][0])
        pred_idx = 1 if score >= 0.5 else 0
        confidence = score if pred_idx == 1 else (1 - score)
    else:
        # multi-class softmax output
        pred_idx = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))

    label = class_names[pred_idx]
    return label, confidence


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def run_demo(
    camera: int = 0,
    video: str | None = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    nms_threshold: float = DEFAULT_NMS_THRESHOLD,
    model_dir: str | Path | None = None,
    spoof_model_path: str | Path | None = None,
) -> None:
    """
    Open a camera or video file and display live face detection results.

    Parameters
    ----------
    camera : int
        OpenCV camera device index (used when *video* is None).
    video : str | None
        Path to a video file.  If provided, *camera* is ignored.
    conf_threshold : float
        Minimum detection confidence to display.
    nms_threshold : float
        IoU threshold for NMS.
    model_dir : str | Path | None
        Override the default model directory (useful for testing).
    spoof_model_path : str | Path | None
        Path to the anti-spoofing Keras model.  Defaults to
        ``anti_spoofing_module_MobileNetV2/model/best_fasd_mobilenetv2_model.keras``.
    """
    # ---- Initialise detector ------------------------------------------------
    detector_kwargs: dict = {
        "conf_threshold": conf_threshold,
        "nms_threshold": nms_threshold,
        "auto_download": True,
    }
    if model_dir is not None:
        detector_kwargs["model_dir"] = model_dir

    print("[demo] Initialising FaceDetector …")
    detector = FaceDetector(**detector_kwargs)
    print(f"[demo] {detector}")

    # ---- Load anti-spoofing model -------------------------------------------
    _spoof_path = Path(spoof_model_path) if spoof_model_path else _DEFAULT_SPOOF_MODEL_PATH

    print(f"[demo] Loading anti-spoofing model from: {_spoof_path}")
    # SavedModel directory and new-format .keras (zip) both load fine with
    # tf.keras.models.load_model in Keras 3.  Legacy HDF5 files (TF2 era)
    # contain TFOpLambda layers that Keras 3 cannot deserialise — those must
    # be resaved as SavedModel from the training notebook first.
    spoof_model = tf.keras.models.load_model(str(_spoof_path), compile=False)
    print("[demo] Anti-spoofing model loaded.")

    # ---- Open video source --------------------------------------------------
    source = video if video else camera
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[demo] ERROR: Could not open source: {source}", file=sys.stderr)
        sys.exit(1)

    print(f"[demo] Opened source: {source!r}")
    print("[demo] Controls:  Q / ESC = quit   |   S = save snapshot")

    # ---- FPS ring buffer ----------------------------------------------------
    frame_times: list[float] = []
    fps_display: float = 0.0

    # ---- Main loop ----------------------------------------------------------
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            if video:
                print("[demo] End of video.")
            else:
                print("[demo] Camera read failed.")
            break

        t_start = time.perf_counter()

        # Convert BGR → RGB for the detector
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run detection
        bboxes, scores, landmarks = detector.detect(frame_rgb)

        # Anti-spoofing: classify each detected face crop
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            face_rgb = frame_rgb[y1:y2, x1:x2]
            if face_rgb.size == 0:
                continue
            label, confidence = predict_real_or_spoof(face_rgb, spoof_model, _SPOOF_CLASS_NAMES)
            color = (0, 255, 0) if label == "real" else (0, 0, 255)
            cv2.putText(
                frame_bgr,
                f"{label}: {confidence:.2%}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        t_end = time.perf_counter()

        # Update FPS (smoothed over last N frames)
        frame_times.append(t_end - t_start)
        if len(frame_times) > _FPS_SMOOTH:
            frame_times.pop(0)
        if frame_times:
            fps_display = 1.0 / (sum(frame_times) / len(frame_times))

        # Draw results (on the BGR frame — OpenCV convention)
        draw_detections(frame_bgr, bboxes, scores, landmarks)
        draw_fps(frame_bgr, fps_display)

        # Face count overlay
        cv2.putText(
            frame_bgr,
            f"Faces: {len(bboxes)}",
            (10, 60),
            _FPS_FONT, _FPS_SCALE, _FPS_COLOR, _FPS_THICKNESS,
            cv2.LINE_AA,
        )

        cv2.imshow("Face Detection Demo — CL04-G01", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            print("[demo] Quit requested.")
            break
        elif key in (ord("s"), ord("S")):
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"demo_snapshot_{ts}.jpg"
            cv2.imwrite(fname, frame_bgr)
            print(f"[demo] Snapshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[demo] Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live face detection demo using InsightFace SCRFD (det_10g.onnx).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        metavar="INDEX",
        help="Webcam device index.",
    )
    src.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a video file (overrides --camera).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        metavar="FLOAT",
        help="Minimum confidence threshold (0–1).",
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=DEFAULT_NMS_THRESHOLD,
        metavar="FLOAT",
        help="NMS IoU threshold (0–1).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override the default model directory.",
    )
    parser.add_argument(
        "--spoof-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to the anti-spoofing Keras model (.keras). Defaults to best_fasd_mobilenetv2_model.keras.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        camera=args.camera,
        video=args.video,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        model_dir=args.model_dir,
        spoof_model_path=args.spoof_model,
    )
