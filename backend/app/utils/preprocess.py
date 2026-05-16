from __future__ import annotations

import io

import numpy as np
from PIL import Image

def load_image_from_bytes(contents: bytes) -> Image.Image:
	image = Image.open(io.BytesIO(contents))
	return image.convert("RGB")

def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
	"""Crop a single normalised (x, y, w, h) box from a PIL image."""
	x, y, w, h = box
	img_w, img_h = image.size
	x1 = int(max(0, x * img_w))
	y1 = int(max(0, y * img_h))
	x2 = int(min(img_w, (x + w) * img_w))
	y2 = int(min(img_h, (y + h) * img_h))
	x2 = max(x2, x1 + 1)
	y2 = max(y2, y1 + 1)
	return image.crop((x1, y1, x2, y2))


def crop_faces(
	image: Image.Image,
	bboxes: list[tuple[float, float, float, float]],
	resize: tuple[int, int] | None = None,
) -> list[Image.Image]:
	"""
	Crop all bboxes from image in one pass.

	Parameters
	----------
	image   : PIL.Image.Image
	bboxes  : list of (x, y, w, h) in normalised [0, 1] coordinates
	resize  : (width, height) to resize each crop; None = no resize

	Returns
	-------
	list[PIL.Image.Image]  — one crop per bbox, same order
	"""
	img_w, img_h = image.size
	img_np = np.array(image)
	crops: list[Image.Image] = []

	for x, y, w, h in bboxes:
		x1 = int(max(0, x * img_w))
		y1 = int(max(0, y * img_h))
		x2 = int(min(img_w, (x + w) * img_w))
		y2 = int(min(img_h, (y + h) * img_h))
		crop = Image.fromarray(img_np[y1:y2, x1:x2])
		if resize is not None:
			crop = crop.resize(resize, Image.BILINEAR)
		crops.append(crop)

	return crops



