from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.base_service import BaseService


class AntiSpoofingService(BaseService):
	def __init__(
		self,
		use_triton: bool | None = None,
		triton_url: str | None = None,
		weights_dir: str | Path | None = None,
		model_path: str | Path | None = None,
	) -> None:
		super().__init__(
			"anti_spoofing",
			use_triton=use_triton,
			triton_url=triton_url,
			weights_dir=weights_dir,
			model_path=model_path,
		)

	# TODO: fill in once model is ready
	def preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
		raise NotImplementedError

	def postprocess(self, outputs: dict[str, np.ndarray]) -> dict:
		raise NotImplementedError

	def predict(self, face_images: Image.Image | list[Image.Image]) -> list[dict]:
		"""
		Output per face:
		  {
		    "label":      "real" | "spoof",
		    "confidence": float          # 0.0 – 1.0
		  }
		Returns a list even for a single image input.
		"""
		if isinstance(face_images, Image.Image):
			face_images = [face_images]
		# this is a dummy implementation; replace with actual model inference but follow the same output format
		return [{"label": "real", "confidence": 0.65} for _ in face_images]
