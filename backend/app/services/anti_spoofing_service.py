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
		self.INPUT_SIZE = (224, 224)
		self.LABELS = ["real", "spoof"]

	def preprocess(self, image: np.ndarray | list[np.ndarray]) -> dict[str, np.ndarray]:
		images = image if isinstance(image, list) else [image]
		input_tensors = []
		for image_item in images:
			resized = Image.fromarray(image_item).resize(self.INPUT_SIZE, Image.BILINEAR)
			# ONNX model includes MobileNetV2 preprocess_input; keep RGB in [0, 255].
			arr = np.asarray(resized, dtype=np.float32)
			input_tensors.append(arr)

		input_name = self._input_metadata[0]["name"] if self._input_metadata else "input"
		return {input_name: np.stack(input_tensors, axis=0).astype(np.float32)}

	def postprocess(self, outputs: dict[str, np.ndarray]) -> list[dict]:
		probs = next(iter(outputs.values())).astype(np.float32)
		if probs.ndim == 1:
			probs = probs[np.newaxis, :]

		results: list[dict] = []
		for row in probs:
			top_idx = int(np.argmax(row))
			label = self.LABELS[top_idx] if top_idx < len(self.LABELS) else str(top_idx)
			results.append(
				{
					"label": label,
					"confidence": float(row[top_idx]),
				}
			)
		return results

	def predict(
		self, face_images: Image.Image | list[Image.Image]
	) -> dict | list[dict]:
		"""
		Output per face:
		  {
		    "label":      "real" | "spoof",
		    "confidence": float          # 0.0 – 1.0
		  }

		Returns a dict for a single image, or a list for multiple images.
		"""
		single_image = isinstance(face_images, Image.Image)
		if single_image:
			face_images = [face_images]

		self._ensure_loaded()
		images = [
			np.asarray(face_image.convert("RGB"), dtype=np.uint8)
			for face_image in face_images
		]
		raw_outputs = self._infer(self.preprocess(images))
		results = self.postprocess(raw_outputs)
		return results[0] if single_image else results
