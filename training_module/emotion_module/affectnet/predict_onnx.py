import sys
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
from PIL import Image

from model import get_convnext_tiny_norm_stats
from model import get_mobilenet_norm_stats

ONNX_PATH = "emotion.onnx"
TRAIN_DIR = "AffectNet/Train"
IMG_SIZE = 224
NORM_MEAN, NORM_STD = get_convnext_tiny_norm_stats()
EMOTIONS = datasets.ImageFolder(TRAIN_DIR).classes

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def _get_providers() -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def predict(image_path: str, onnx_path: str) -> None:
    session = ort.InferenceSession(onnx_path, providers=_get_providers())
    input_name = session.get_inputs()[0].name

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    input_data = tensor.numpy().astype(np.float32)

    logits = session.run(None, {input_name: input_data})[0]
    probs = softmax(logits).squeeze(0)

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    print(f"\nResult: {EMOTIONS[top_idx].upper()} ({top_prob*100:.1f}%)\n")
    for emotion, prob in zip(EMOTIONS, probs):
        bar = "█" * int(float(prob) * 30)
        print(f"  {emotion:<12} {prob*100:5.1f}%  {bar}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_onnx.py <image_path> [onnx_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    if len(sys.argv) > 2:
        ONNX_PATH = sys.argv[2]

    predict(image_path, ONNX_PATH)
