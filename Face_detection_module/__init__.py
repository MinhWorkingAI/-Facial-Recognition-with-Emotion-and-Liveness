"""
Face_detection_module
=====================
InsightFace SCRFD-based face detection module for COS40007 Group CL04-G01.

Public API
----------
::

    from Face_detection_module import FaceDetector

    detector = FaceDetector()
    bboxes, scores, landmarks = detector.detect(frame_rgb)

See ``face_detection.FaceDetector`` for full parameter and return-value
documentation.
"""

from .face_detection import (
    FaceDetector,
    DEFAULT_MODEL_DIR,
    DEFAULT_INPUT_SIZE,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_NMS_THRESHOLD,
)

__all__ = [
    "FaceDetector",
    "DEFAULT_MODEL_DIR",
    "DEFAULT_INPUT_SIZE",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_NMS_THRESHOLD",
]

__version__ = "0.1.0"
