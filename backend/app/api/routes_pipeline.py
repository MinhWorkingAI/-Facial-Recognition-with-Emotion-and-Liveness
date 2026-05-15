from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.frame_schema import (
    AntiSpoofingResult,
    DetectedFace,
    EmotionResult,
    FaceAnalysis,
    FrameAnalysisResponse,
    NormalizedBox,
    RecognitionResult,
)
from app.services.anti_spoofing_service import AntiSpoofingService
from app.services.emotion_service import EmotionService
from app.services.face_detection_service import FaceDetectionService
from app.services.verification_service import VerificationService
from app.utils.preprocess import load_image_from_bytes


router = APIRouter()
face_detection = FaceDetectionService()
emotion_service = EmotionService()
anti_spoofing_service = AntiSpoofingService()
verification_service = VerificationService()


@router.post("/frame", response_model=FrameAnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)) -> FrameAnalysisResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image size")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    detections = face_detection.detect(image)
    face_results: list[FaceAnalysis] = []

    for detection in detections:
        bbox = detection["bbox"]
        crop = detection["crop"]
        crop_width, crop_height = crop.size

        emotion = emotion_service.predict(crop)
        anti_spoofing = anti_spoofing_service.predict(crop)
        recognition = verification_service.verify(crop)

        face_results.append(
            FaceAnalysis(
                face=DetectedFace(
                    bbox=NormalizedBox(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3]),
                    detection_confidence=detection["confidence"],
                    crop_width=crop_width,
                    crop_height=crop_height,
                ),
                emotion=EmotionResult(**emotion),
                anti_spoofing=AntiSpoofingResult(**anti_spoofing),
                recognition=RecognitionResult(**recognition),
            )
        )

    return FrameAnalysisResponse(image_width=width, image_height=height, faces=face_results)
