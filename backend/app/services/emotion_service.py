from PIL import Image

from app.services.base_service import BaseService


class EmotionService(BaseService):
	def __init__(self) -> None:
		super().__init__("emotion")

	def predict(self, face_image: Image.Image) -> dict:
		return {"label": "neutral", "confidence": 0.7}
