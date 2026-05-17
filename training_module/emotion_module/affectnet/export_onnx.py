import sys
import torch

from model import build_model

MODEL_PATH = "emotion.pth"
ONNX_PATH = "emotion.onnx"
IMG_SIZE = 224


def _infer_num_classes(state_dict: dict) -> int:
    for key in ("classifier.4.weight", "classifier.4.bias"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("Unable to infer num_classes from checkpoint.")


def export_onnx(model_path: str, onnx_path: str) -> None:
    state = torch.load(model_path, map_location="cpu")
    num_classes = _infer_num_classes(state)
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        ONNX_PATH = sys.argv[2]

    export_onnx(MODEL_PATH, ONNX_PATH)
    print(f"Exported ONNX to: {ONNX_PATH}")
