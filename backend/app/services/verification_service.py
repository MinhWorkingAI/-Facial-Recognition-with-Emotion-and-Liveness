from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.services.base_service import BaseService


class VerificationService(BaseService):
	def __init__(
		self,
		use_triton: bool | None = None,
		triton_url: str | None = None,
		weights_dir: str | Path | None = None,
		model_path: str | Path | None = None,
		match_threshold: float = 0.6,
	) -> None:
		super().__init__(
			"arcface",
			use_triton=use_triton,
			triton_url=triton_url,
			weights_dir=weights_dir,
			model_path=model_path,
		)
		self.match_threshold = match_threshold
		self._registered_embeddings: dict[str, np.ndarray] = {}
		self.INPUT_SIZE = (112, 112)
		self.CHUNK_SIZE = 32

	def preprocess(self, image: np.ndarray | list[np.ndarray]) -> dict[str, np.ndarray]:
		images = image if isinstance(image, list) else [image]
		input_tensors = []
		for image_item in images:
			resized = Image.fromarray(image_item).resize(self.INPUT_SIZE, Image.BILINEAR)
			input_tensor = np.asarray(resized, dtype=np.float32)
			input_tensor = (input_tensor - 127.5) / 128.0
			input_tensors.append(input_tensor)

		input_name = self._input_metadata[0]["name"] if self._input_metadata else "input_1"
		return {input_name: np.stack(input_tensors, axis=0).astype(np.float32)}

	def postprocess(self, outputs: dict[str, np.ndarray]) -> np.ndarray:
		return self._normalize_outputs(outputs)[0]

	def verify(self, face_images: Image.Image | list[Image.Image]) -> list[dict]:
		"""
		Output per face:
		  {
		    "employee_id":   str | None,  # DB employee ID; None if unrecognised
		    "employee_name": str,         # display name; "unknown" if unrecognised
		    "confidence":    float,       # cosine similarity score 0.0 – 1.0
		    "matched":       bool         # True if confidence >= verification threshold
		  }
		Returns a list even for a single image input.
		"""
		if isinstance(face_images, Image.Image):
			face_images = [face_images]

		results: list[dict] = []
		for embedding in self._extract_embeddings(face_images):
			best_person_id, best_score = self._find_best_match(embedding)
			matched = best_person_id is not None and best_score >= self.match_threshold
			employee_name = best_person_id if matched else "unknown"

			results.append(
				{
					"employee_id": best_person_id if matched else None,
					"employee_name": employee_name,
					"label": employee_name,
					"confidence": float(best_score),
					"matched": matched,
				}
			)
		return results

	def register(self, face_image: Image.Image, person_id: str) -> dict:
		"""
		Output:
		  {
		    "person_id": str,
		    "status":    "registered" | "updated"
		  }
		"""
		status = "updated" if person_id in self._registered_embeddings else "registered"
		self._registered_embeddings[person_id] = self._extract_embeddings([face_image])[0]
		return {"person_id": person_id, "status": status}

	def _extract_embeddings(self, face_images: list[Image.Image]) -> list[np.ndarray]:
		if not face_images:
			return []

		self._ensure_loaded()
		embeddings: list[np.ndarray] = []
		input_name = self._input_metadata[0]["name"]

		for start in range(0, len(face_images), self.CHUNK_SIZE):
			chunk = face_images[start:start + self.CHUNK_SIZE]
			images = [
				np.asarray(face_image.convert("RGB"), dtype=np.uint8)
				for face_image in chunk
			]
			raw_outputs = self._infer(self.preprocess(images))
			embeddings.extend(self._normalize_outputs(raw_outputs))

		return embeddings

	def _normalize_outputs(self, outputs: dict[str, np.ndarray]) -> list[np.ndarray]:
		embeddings = next(iter(outputs.values())).astype(np.float32)
		if embeddings.ndim == 1:
			embeddings = embeddings[np.newaxis, :]

		norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
		if np.any(norms == 0):
			raise ValueError("ArcFace model returned a zero-norm embedding.")

		normalized = embeddings / norms
		return [embedding for embedding in normalized]

	def _find_best_match(self, embedding: np.ndarray) -> tuple[str | None, float]:
		if not self._registered_embeddings:
			return None, 0.0

		best_person_id: str | None = None
		best_score = -1.0
		for person_id, known_embedding in self._registered_embeddings.items():
			score = float(np.dot(embedding, known_embedding))
			if score > best_score:
				best_person_id = person_id
				best_score = score

		return best_person_id, max(best_score, 0.0)
