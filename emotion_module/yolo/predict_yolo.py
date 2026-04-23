import sys
from ultralytics import YOLO

MODEL_PATH = "emotion_yolo/train/weights/best.pt"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_yolo.py <image_path>")
        sys.exit(1)

    model      = YOLO(MODEL_PATH)
    image_path = sys.argv[1]

    results = model.predict(source=image_path, imgsz=640, conf=0.25)

    for result in results:
        for box in result.boxes:
            cls   = int(box.cls)
            conf  = float(box.conf)
            label = result.names[cls]
            print(f"Detected: {label} ({conf*100:.1f}%)")

        result.save(filename="prediction_output.jpg")
        print("Saved: prediction_output.jpg")