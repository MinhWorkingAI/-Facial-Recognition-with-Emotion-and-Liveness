from PIL import Image

from app.services.base_service import BaseService


class AntiSpoofingService(BaseService):
	def __init__(self) -> None:
		super().__init__("anti_spoofing")

	def predict(self, face_image: Image.Image) -> dict:
		return {"label": "real", "confidence": 0.65}
