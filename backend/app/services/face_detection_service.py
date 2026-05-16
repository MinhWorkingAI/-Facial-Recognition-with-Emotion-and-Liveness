from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.base_service import BaseService
from app.utils.preprocess import crop_faces


class FaceDetectionService(BaseService):
	def __init__(
		self,
		use_triton: bool | None = None,
		triton_url: str | None = None,
		weights_dir: str | Path | None = None,
		model_path: str | Path | None = None,
	) -> None:
		super().__init__(
			"face_detection",
			use_triton=use_triton,
			triton_url=triton_url,
			weights_dir=weights_dir,
			model_path=model_path,
		)

	# TODO: fill in once model is ready
	def preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
		raise NotImplementedError

	def postprocess(self, outputs: dict[str, np.ndarray]) -> list[dict]:
		raise NotImplementedError

	def detect(self, image: Image.Image) -> list[dict]:
		"""
		Output per face:
		  {
		    "bbox":       (x, y, w, h),  # normalised [0, 1] coordinates
		    "confidence": float,          # detection score 0.0 – 1.0
		    "crop":       Image.Image     # cropped face region
		  }
		"""

		# this is a dummy implementation; replace with actual model inference but follow the same output format
		# WARNING: Must use crop_faces util to ensure the crop coordinates are consistent with the dummy bbox format, otherwise downstream models will break when we switch to real face detection outputs
		box = (0.3, 0.3, 0.4, 0.4)
		crops = crop_faces(image, [box], resize=(512, 512))
		return [
			{
				"bbox": box,
				"confidence": 0.6,
				"crop": crops[0],
			}
		]
