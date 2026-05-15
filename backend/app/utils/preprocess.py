import io
import base64

from PIL import Image


def load_image_from_bytes(contents: bytes) -> Image.Image:
	image = Image.open(io.BytesIO(contents))
	return image.convert("RGB")


def clamp_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
	x = max(0.0, min(1.0, x))
	y = max(0.0, min(1.0, y))
	w = max(0.0, min(1.0, w))
	h = max(0.0, min(1.0, h))
	return x, y, w, h


def crop_image(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
	x, y, w, h = clamp_box(*box)
	width, height = image.size
	left = int(x * width)
	top = int(y * height)
	right = int((x + w) * width)
	bottom = int((y + h) * height)
	if right <= left:
		right = min(width, left + 1)
	if bottom <= top:
		bottom = min(height, top + 1)
	return image.crop((left, top, right, bottom))


