from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.verification_schema import RecognitionResponse, RecognitionResult, RegisterResponse
from app.services.verification_service import VerificationService
from app.utils.preprocess import load_image_from_bytes


router = APIRouter()
verification_service = VerificationService()


@router.post("/verify", response_model=RecognitionResponse)
async def verify_face(file: UploadFile = File(...)) -> RecognitionResponse:
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

    result = verification_service.verify(image)[0]
    return RecognitionResponse(recognition=RecognitionResult(**result))


@router.post("/register", response_model=RegisterResponse)
async def register_face(
    person_id: str = Form(...),
    file: UploadFile = File(...),
) -> RegisterResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    result = verification_service.register(image, person_id)
    return RegisterResponse(
        person_id=result["person_id"],
        status=result["status"],
        message="Face embedding stored in memory.",
    )
