from pathlib import Path
from ultralytics import YOLO

MODEL_PATH   = "yolo11s.pt"
DATA_YAML    = "YOLO_format/data.yaml"
EPOCHS       = 25
IMG_SIZE     = 160
PROJECT_NAME = "emotion_yolo"

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        project=PROJECT_NAME,
        plots=True,
    )

    best = Path(model.trainer.best)
    print(f"\nTraining complete. Best model: {best}")

    eval_model = YOLO(best)
    for split in ["val", "test"]:
        print(f"\n--- {split.upper()} ---")
        metrics = eval_model.val(
            data=DATA_YAML,
            split=split,
            imgsz=IMG_SIZE,
            plots=True,
        )
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")