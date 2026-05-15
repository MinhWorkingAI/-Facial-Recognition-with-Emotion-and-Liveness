from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.frame_schema import DetectionResponse, DetectedFace, NormalizedBox
from app.services.face_detection_service import FaceDetectionService
from app.utils.preprocess import load_image_from_bytes


router = APIRouter()
face_detection = FaceDetectionService()


@router.post("/detect", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)) -> DetectionResponse:
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
    faces: list[DetectedFace] = []

    for detection in detections:
        bbox = detection["bbox"]
        crop = detection["crop"]
        crop_width, crop_height = crop.size

        faces.append(
            DetectedFace(
                bbox=NormalizedBox(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3]),
                detection_confidence=detection["confidence"],
                crop_width=crop_width,
                crop_height=crop_height,
                crop_base64=None,
            )
        )

    return DetectionResponse(image_width=width, image_height=height, faces=faces)
