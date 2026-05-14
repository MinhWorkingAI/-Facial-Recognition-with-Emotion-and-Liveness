"""
face_detection.py
=================
Core face-detection module for the COS40007 Group CL04-G01 pipeline.

This file is the **only import surface** other modules should use.  Camera
capture, display, or GUI logic must NOT live here — this file is a pure
inference engine.

Architecture
------------
Internally this module uses InsightFace's SCRFD detector (``det_10g.onnx``
from the buffalo_l pack).  SCRFD (Sample and Computation Redistribution for
Efficient Face Detection) is a single-shot, anchor-free detector that outputs
multi-scale feature maps at strides 8, 16, and 32.

The ONNX session is configured to run on CUDA (GPU) with an automatic
fallback to CPU if no compatible GPU is detected.

Public API
----------
::

    from Face_detection_module.face_detection import FaceDetector

    detector = FaceDetector()                      # default model dir
    bboxes, scores, landmarks = detector.detect(frame_rgb)

See ``FaceDetector`` class docstring for full details.

Tensor contracts
----------------
**Input to ``FaceDetector.detect``**

+----------+--------------+----------------------------------------+
| Argument | Shape        | Description                            |
+==========+==============+========================================+
| image    | (H, W, 3)    | uint8 **or** float32 NumPy array.      |
|          |              | Channel order: **RGB**.                |
|          |              | Any spatial resolution is accepted;   |
|          |              | the image is letterbox-resized to the  |
|          |              | model input size (default 640×640).    |
+----------+--------------+----------------------------------------+

**Output of ``FaceDetector.detect``**

+------------+------------------+-------------------------------------------+
| Return val | Shape            | Description                               |
+============+==================+===========================================+
| bboxes     | (N, 4) float32   | Bounding boxes in **pixel coordinates**   |
|            |                  | of the *original* (un-resized) image.     |
|            |                  | Format: [x1, y1, x2, y2].                |
+------------+------------------+-------------------------------------------+
| scores     | (N,) float32     | Detection confidence in [0, 1].           |
+------------+------------------+-------------------------------------------+
| landmarks  | (N, 5, 2) float32| Five facial keypoints per face in pixel  |
|            |                  | coordinates of the original image.        |
|            |                  | Order: left eye, right eye, nose tip,    |
|            |                  | left mouth corner, right mouth corner.   |
+------------+------------------+-------------------------------------------+

When no face is detected all three arrays have shape (0, …) with the correct
dtype so callers can safely iterate without special-casing ``None``.

**Internal ONNX tensor contract (det_10g.onnx / SCRFD-10GF)**

Input node ``"input.1"``

+-------+---------------------------------+----------------------------------+
| Shape | (1, 3, input_size, input_size)  | float32                          |
+-------+---------------------------------+----------------------------------+
| Range | Normalised with ImageNet stats: | mean=[127.5,127.5,127.5]         |
|       |                                 | std =[128.0,128.0,128.0]         |
+-------+---------------------------------+----------------------------------+

Output nodes (InsightFace SCRFD-10GF, 3 FPN strides: 8, 16, 32)

+----------------------------------+---------------------+-------------------+
| Node name (pattern)              | Shape               | Meaning           |
+==================================+=====================+===================+
| score_8 / score_16 / score_32    | (1, A*H*W, 1)       | Objectness logit  |
+----------------------------------+---------------------+-------------------+
| bbox_8 / bbox_16 / bbox_32       | (1, A*H*W, 4)       | BBox deltas (ltrb)|
+----------------------------------+---------------------+-------------------+
| kps_8 / kps_16 / kps_32          | (1, A*H*W, 10)      | KP deltas (xy×5)  |
+----------------------------------+---------------------+-------------------+

A = number of anchors per cell (2 for SCRFD-10GF).
The InsightFace SCRFD wrapper decodes these automatically; we rely on it
rather than re-implementing the anchor grid from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[face_detection] %(levelname)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Default configuration constants
# ---------------------------------------------------------------------------

#: Default root for model artefacts (relative to *this file*).
DEFAULT_MODEL_DIR: Path = Path(__file__).parent / "model"

#: Model sub-directory created by buffalo_l unpacking.
BUFFALO_SUBDIR: str = "buffalo_l"

#: ONNX detector file within the buffalo_l directory.
DETECTOR_FILE: str = "det_10g.onnx"

#: Input spatial resolution fed to SCRFD (must be divisible by 32).
DEFAULT_INPUT_SIZE: int = 640

#: Minimum objectness score to keep a detection.
DEFAULT_CONF_THRESHOLD: float = 0.7

#: IoU threshold for Non-Maximum Suppression.
DEFAULT_NMS_THRESHOLD: float = 0.4


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Wrapper around InsightFace's SCRFD detector (``det_10g.onnx``).

    The detector is loaded once at construction time and can be called
    repeatedly on new frames.  The underlying ONNX session runs on GPU
    (CUDA) when available, otherwise falls back to CPU.

    Parameters
    ----------
    model_dir : str | Path, optional
        Root directory that contains ``buffalo_l/det_10g.onnx``.
        Defaults to ``<this_file_parent>/model``.
    input_size : int, optional
        Square input resolution for the SCRFD model (default 640).
        Must be a multiple of 32.
    conf_threshold : float, optional
        Minimum confidence score to keep a detection (default 0.7).
    nms_threshold : float, optional
        IoU threshold for NMS (default 0.4).
    auto_download : bool, optional
        If True (default), trigger :func:`download_model.ensure_model` when
        the ONNX file is not found, so the user never needs to run the
        downloader manually.

    Example
    -------
    ::

        import cv2
        from Face_detection_module.face_detection import FaceDetector

        detector = FaceDetector()
        frame_bgr = cv2.imread("photo.jpg")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        bboxes, scores, landmarks = detector.detect(frame_rgb)
        # bboxes    : np.ndarray, shape (N, 4),    dtype float32, [x1,y1,x2,y2]
        # scores    : np.ndarray, shape (N,),       dtype float32
        # landmarks : np.ndarray, shape (N, 5, 2), dtype float32
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        input_size: int = DEFAULT_INPUT_SIZE,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        auto_download: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self._detector = self._load_detector(auto_download)

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _load_detector(self, auto_download: bool):
        """
        Load the SCRFD ONNX model via InsightFace's model zoo.

        Steps
        -----
        1. Check that ``det_10g.onnx`` exists; trigger download if needed.
        2. Determine the best available ONNX execution provider (CUDA → CPU).
        3. Construct an InsightFace ``SCRFD`` model object, call ``prepare``
           with the chosen provider context, and return it.

        Parameters
        ----------
        auto_download : bool
            Whether to call :func:`download_model.ensure_model` on miss.

        Returns
        -------
        insightface.model_zoo.retinaface.SCRFD
            Prepared detector instance.
        """
        detector_path = self.model_dir / BUFFALO_SUBDIR / DETECTOR_FILE

        if not detector_path.exists():
            if auto_download:
                logger.info(
                    "Detector ONNX not found at %s — triggering download …",
                    detector_path,
                )
                from .download_model import ensure_model  # lazy import
                ensure_model(self.model_dir)
            else:
                raise FileNotFoundError(
                    f"Detector ONNX not found: {detector_path}\n"
                    "Set auto_download=True or run download_model.py manually."
                )

        # ---- Choose ONNX execution provider --------------------------------
        provider = self._select_provider()

        # ---- Load via InsightFace SCRFD ------------------------------------
        try:
            from insightface.model_zoo import model_zoo
        except ImportError as exc:
            raise ImportError(
                "insightface is not installed. "
                "Install it with:  pip install insightface"
            ) from exc

        logger.info("Loading SCRFD model from %s …", detector_path)
        detector = model_zoo.get_model(str(detector_path))

        # ``prepare`` configures the internal ONNX session.
        # ctx_id  0  = first CUDA device  (GPU)
        # ctx_id -1  = CPU
        ctx_id = 0 if provider == "CUDAExecutionProvider" else -1
        detector.prepare(
            ctx_id=ctx_id,
            input_size=(self.input_size, self.input_size),
            det_thresh=self.conf_threshold,
        )

        logger.info(
            "Detector ready  |  provider=%s  |  input=%dx%d  |  "
            "conf_thresh=%.2f  |  nms_thresh=%.2f",
            provider,
            self.input_size,
            self.input_size,
            self.conf_threshold,
            self.nms_threshold,
        )
        return detector

    @staticmethod
    def _select_provider() -> str:
        """
        Return the best available ONNX Runtime execution provider.

        Tries ``CUDAExecutionProvider`` first (requires ``onnxruntime-gpu``
        and a compatible CUDA installation).  Falls back to
        ``CPUExecutionProvider`` if CUDA is not available.

        Returns
        -------
        str
            Either ``"CUDAExecutionProvider"`` or ``"CPUExecutionProvider"``.
        """
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                logger.info("CUDA is available — using GPU inference.")
                return "CUDAExecutionProvider"
            else:
                logger.info("CUDA not available — falling back to CPU.")
                return "CPUExecutionProvider"
        except ImportError:
            logger.warning(
                "onnxruntime not found; InsightFace will manage the session "
                "internally.  Install onnxruntime-gpu for GPU support."
            )
            return "CPUExecutionProvider"

    # ------------------------------------------------------------------
    # Public: inference
    # ------------------------------------------------------------------

    def detect(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run face detection on a single RGB image.

        This is the **primary public method**.  It performs:

        1. Input validation and uint8 normalisation.
        2. Invocation of the InsightFace SCRFD ``detect`` method, which
           internally handles letterbox resize, ONNX forward pass, anchor
           decoding, sigmoid activation, and NMS.
        3. Re-application of the caller's ``nms_threshold`` (SCRFD uses a
           fixed IoU; we post-filter with our own threshold via a second
           NMS pass when needed).
        4. Reshaping of raw keypoint output from (N, 10) → (N, 5, 2).
        5. Return three separate NumPy arrays.

        Parameters
        ----------
        image : np.ndarray
            Shape ``(H, W, 3)``, channel order **RGB**.
            Accepts both ``uint8`` (0–255) and ``float32`` (0.0–255.0)
            arrays.  Any spatial resolution is accepted.

        Returns
        -------
        bboxes : np.ndarray
            Shape ``(N, 4)``, dtype ``float32``.
            Bounding boxes as ``[x1, y1, x2, y2]`` in **original image**
            pixel coordinates (i.e. after undoing the internal resize).

        scores : np.ndarray
            Shape ``(N,)``, dtype ``float32``.
            Confidence scores in the range ``[0, 1]`` for each detected
            face, in the same order as ``bboxes`` and ``landmarks``.

        landmarks : np.ndarray
            Shape ``(N, 5, 2)``, dtype ``float32``.
            Five facial keypoints per face in **original image** pixel
            coordinates.  Keypoint order (InsightFace convention):

            * ``[0]`` — left eye
            * ``[1]`` — right eye
            * ``[2]`` — nose tip
            * ``[3]`` — left mouth corner
            * ``[4]`` — right mouth corner

        Notes
        -----
        * When *N = 0* (no detections), all arrays have shape ``(0, …)``
          so callers can iterate unconditionally.
        * This method is **not** thread-safe by default; construct one
          ``FaceDetector`` per thread if parallel inference is required.

        Raises
        ------
        ValueError
            If *image* does not have shape ``(H, W, 3)``.
        TypeError
            If *image* is not a NumPy array.
        """
        # ---- Input validation -----------------------------------------------
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a NumPy ndarray, got {type(image).__name__}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have shape (H, W, 3), got {image.shape}"
            )

        # Ensure uint8 for InsightFace internals
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # ---- Run SCRFD detection -------------------------------------------
        # InsightFace's SCRFD.detect returns:
        #   bboxes     : (N, 5)  — [x1, y1, x2, y2, score]
        #   keypoints  : (N, 5, 2) | None
        #
        # The max_num=0 argument means "return all detections".
        raw_bboxes, raw_kps = self._detector.detect(
            image,
            max_num=0,
            metric="default",
        )

        # ---- Handle empty results ------------------------------------------
        empty_bboxes = np.zeros((0, 4), dtype=np.float32)
        empty_scores = np.zeros((0,), dtype=np.float32)
        empty_landmarks = np.zeros((0, 5, 2), dtype=np.float32)

        if raw_bboxes is None or len(raw_bboxes) == 0:
            return empty_bboxes, empty_scores, empty_landmarks

        # ---- Unpack combined bbox+score array ------------------------------
        # SCRFD returns bboxes as (N, 5): columns 0-3 are x1y1x2y2, col 4 is score
        bboxes = raw_bboxes[:, :4].astype(np.float32)
        scores = raw_bboxes[:, 4].astype(np.float32)

        # ---- Apply caller's NMS threshold (secondary filter) ---------------
        keep = self._nms(bboxes, scores, self.nms_threshold)
        bboxes = bboxes[keep]
        scores = scores[keep]

        # ---- Landmarks ---------------------------------------------------------
        if raw_kps is not None and len(raw_kps) > 0:
            # InsightFace already returns (N, 5, 2) for SCRFD.
            kps = raw_kps[keep].astype(np.float32)
            # Ensure correct shape even for single-face case.
            if kps.ndim == 2:
                kps = kps.reshape(-1, 5, 2)
        else:
            kps = np.zeros((len(bboxes), 5, 2), dtype=np.float32)

        return bboxes, scores, kps

    # ------------------------------------------------------------------
    # Private: NMS helper
    # ------------------------------------------------------------------

    @staticmethod
    def _nms(
        bboxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """
        Pure-NumPy Non-Maximum Suppression.

        Used as a secondary NMS step to enforce the caller-specified
        ``nms_threshold``.  Boxes are processed in descending score order.

        Parameters
        ----------
        bboxes : np.ndarray
            Shape ``(N, 4)``, format ``[x1, y1, x2, y2]``.
        scores : np.ndarray
            Shape ``(N,)``, confidence scores.
        iou_threshold : float
            Boxes whose IoU with a kept box exceeds this value are
            suppressed.

        Returns
        -------
        np.ndarray
            1-D array of integer indices of the boxes to keep, sorted by
            descending score.
        """
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Intersection
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, ix2 - ix1 + 1) * np.maximum(0.0, iy2 - iy1 + 1)

            # IoU
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int64)

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        """Return the active ONNX execution provider name."""
        return self._select_provider()

    def __repr__(self) -> str:
        return (
            f"FaceDetector("
            f"input_size={self.input_size}, "
            f"conf_threshold={self.conf_threshold}, "
            f"nms_threshold={self.nms_threshold}, "
            f"provider={self.provider})"
        )
