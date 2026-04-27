import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import sys

from model import build_model, get_mobilenet_norm_stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
TRAIN_DIR = "AffectNet/Train"
IMG_SIZE = 96
NORM_MEAN, NORM_STD = get_mobilenet_norm_stats()
EMOTIONS = datasets.ImageFolder(TRAIN_DIR).classes


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    model = build_model(num_classes=len(EMOTIONS), freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    img    = Image.open(sys.argv[1]).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze()

    top_prob, top_idx = probs.max(0)
    print(f"\nResult: {EMOTIONS[top_idx].upper()} ({top_prob*100:.1f}%)\n")
    for emotion, prob in zip(EMOTIONS, probs):
        bar = "█" * int(prob.item() * 30)
        print(f"  {emotion:<12} {prob*100:5.1f}%  {bar}")