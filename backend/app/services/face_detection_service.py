from PIL import Image

from app.services.base_service import BaseService
from app.utils.preprocess import crop_image


class FaceDetectionService(BaseService):
	def __init__(self) -> None:
		super().__init__("face_detection")

	def detect(self, image: Image.Image) -> list[dict]:
		box_w = 0.4
		box_h = 0.4
		box_x = 0.5 - (box_w / 2)
		box_y = 0.5 - (box_h / 2)
		crop = crop_image(image, (box_x, box_y, box_w, box_h))

		return [
			{
				"bbox": (box_x, box_y, box_w, box_h),
				"confidence": 0.6,
				"crop": crop,
			}
		]
