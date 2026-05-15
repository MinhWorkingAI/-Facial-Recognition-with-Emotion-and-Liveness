from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.frame_schema import AntiSpoofingResponse, AntiSpoofingResult
from app.services.anti_spoofing_service import AntiSpoofingService
from app.utils.preprocess import load_image_from_bytes


router = APIRouter()
anti_spoofing_service = AntiSpoofingService()


@router.post("/anti-spoof", response_model=AntiSpoofingResponse)
async def analyze_anti_spoofing(file: UploadFile = File(...)) -> AntiSpoofingResponse:
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

    result = anti_spoofing_service.predict(image)
    return AntiSpoofingResponse(anti_spoofing=AntiSpoofingResult(**result))
