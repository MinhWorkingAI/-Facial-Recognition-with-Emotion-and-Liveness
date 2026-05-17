from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.emotion_schema import EmotionResponse, EmotionResult
from app.services.emotion_service import EmotionService
from app.utils.preprocess import load_image_from_bytes


router = APIRouter()
emotion_service = EmotionService()


@router.post("/", response_model=EmotionResponse)
async def analyze_emotion(file: UploadFile = File(...)) -> EmotionResponse:
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
	result = emotion_service.predict(image)[0]
	return EmotionResponse(emotion=EmotionResult(**result))
