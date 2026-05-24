from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.services.face_detection_service import FaceDetectionService
from app.services.anti_spoofing_service import AntiSpoofingService
from app.services.emotion_service import EmotionService
from app.services.verification_service import VerificationService

from app.schemas.pipeline_schema import FaceResult, InferenceResult

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        liveness_threshold: float = 0.25,
        verification_threshold: float = 0.6,
        use_triton: bool | None = None,
        triton_url: str | None = None,
        weights_dir: str | Path | None = None,
    ) -> None:
        self.face_detection: FaceDetectionService = FaceDetectionService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )
        self.anti_spoofing: AntiSpoofingService = AntiSpoofingService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )
        self.emotion: EmotionService = EmotionService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )
        self.verification: VerificationService = VerificationService(
            use_triton=use_triton,
            triton_url=triton_url,
            weights_dir=weights_dir,
        )

        self.liveness_threshold = liveness_threshold
        self.verification_threshold = verification_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inference(self, image: Image.Image) -> InferenceResult:
        result = InferenceResult()

        detections = self.detect_faces(image)
        if not detections:
            return result

        bboxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]
        face_crops = [d["crop"] for d in detections]

        # Build initial FaceResult list
        face_results = []
        for bbox, score, crop in zip(bboxes, scores, face_crops):
            crop_width, crop_height = crop.size
            face_results.append(
                FaceResult(
                    bbox=bbox,
                    detection_score=score,
                    crop_width=crop_width,
                    crop_height=crop_height,
                )
            )

        # Batch anti-spoofing → filter live faces
        face_results = self.run_anti_spoofing_batch(face_crops, face_results)
        live_idx = [i for i, fr in enumerate(face_results) if fr.is_live]


        if not live_idx:
            result.faces = face_results
            return result

        live_crops = [face_crops[i] for i in live_idx]

        # Batch emotion on live faces only
        live_results = [face_results[i] for i in live_idx]
        live_results = self.run_emotion_batch(live_crops, live_results)

        # Batch verification on live faces only
        live_results = self.run_verification_batch(live_crops, live_results)

        # Merge back and collect attendance
        for i, fr in zip(live_idx, live_results):
            face_results[i] = fr
            if fr.verified and fr.employee_id:
                result.attendance_triggered.append(fr.employee_id)

        result.faces = face_results
        return result

    # ------------------------------------------------------------------
    # Batch pipeline steps
    # ------------------------------------------------------------------

    def detect_faces(self, image: Image.Image) -> list[dict]:
        # Returns list[{"bbox": (x, y, w, h), "confidence": float, "crop": Image.Image}]
        return self.face_detection.detect(image)

    def run_anti_spoofing_batch(
        self, face_crops: list[Image.Image], face_results: list[FaceResult]
    ) -> list[FaceResult]:
        predictions = self.anti_spoofing.predict(face_crops)
        for fr, pred in zip(face_results, predictions):
            fr.liveness_score = pred["confidence"]
            fr.is_live = pred["label"] == "real" and pred["confidence"] >= self.liveness_threshold
        return face_results

    def run_emotion_batch(
        self, face_crops: list[Image.Image], face_results: list[FaceResult]
    ) -> list[FaceResult]:
        predictions = self.emotion.predict(face_crops)
        for fr, pred in zip(face_results, predictions):
            fr.emotion = pred["label"]
            fr.emotion_score = pred["confidence"]
        return face_results

    def run_verification_batch(
        self, face_crops: list[Image.Image], face_results: list[FaceResult]
    ) -> list[FaceResult]:
        predictions = self.verification.verify(face_crops)
        for fr, pred in zip(face_results, predictions):
            fr.employee_id = pred["employee_id"]
            fr.employee_name = pred["employee_name"]
            fr.similarity = pred["confidence"]
            fr.verified = pred["matched"] and pred["confidence"] >= self.verification_threshold
        return face_results
