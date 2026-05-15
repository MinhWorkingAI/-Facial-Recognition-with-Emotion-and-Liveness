from PIL import Image

from app.services.base_service import BaseService


class VerificationService(BaseService):
	def __init__(self) -> None:
		super().__init__("verification")

	def verify(self, face_image: Image.Image) -> dict:
		return {"label": "unknown", "confidence": 0.4, "matched": False}

	def register(self, face_image: Image.Image, person_id: str) -> dict:
		return {"person_id": person_id, "status": "registered"}
