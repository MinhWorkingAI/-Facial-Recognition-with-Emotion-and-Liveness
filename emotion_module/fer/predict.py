import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

from model import build_model

# ── Config ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
IMG_SIZE   = 96
EMOTIONS   = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Transform (same as test transform in preprocessing) ───────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Load model ────────────────────────────────────────────────────────────
def load_model():
    model = build_model(freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ── Predict ───────────────────────────────────────────────────────────────
def predict(image_path, model):
    if not os.path.exists(image_path):
        print(f"Error: file not found — {image_path}")
        return

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # add batch dim

    with torch.no_grad():
        outputs = model(tensor)
        probs   = F.softmax(outputs, dim=1).squeeze()

    top_prob, top_idx = probs.max(0)
    emotion           = EMOTIONS[top_idx.item()]

    print(f"\nImage : {image_path}")
    print(f"Result: {emotion.upper()} ({top_prob.item()*100:.1f}%)")
    print("\nAll probabilities:")
    for i, (emo, prob) in enumerate(zip(EMOTIONS, probs)):
        bar = "█" * int(prob.item() * 30)
        print(f"  {emo:<10} {prob.item()*100:5.1f}%  {bar}")

# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py test_face.jpg")
        sys.exit(1)

    model      = load_model()
    image_path = sys.argv[1]
    predict(image_path, model)