# Backend Developer Guide

This backend is split into small ML services. Each service owns one model task, but all services must keep the same public input/output contracts so the pipeline can combine them.

## Project Structure

```text
backend/
  app/
    api/                  # FastAPI routes
    schemas/              # Pydantic API response models
    services/             # Model and pipeline services
    utils/                # Shared image helpers
    config.py             # Runtime settings
    configs/config.yaml   # Default backend runtime config
  weights/                # Local ONNX model files
  requirements.txt
  test_frame.py
```

## Model Loading

All model services inherit from `BaseService`.

Runtime settings are loaded from:

```text
backend/app/configs/config.yaml
```

You can point to another config file with:

```bash
BACKEND_CONFIG_PATH=/path/to/config.yaml uvicorn app.main:app --reload
```

Environment variables still override YAML values.

## Qdrant Face Database

Face recognition embeddings are stored in Qdrant.

Start Qdrant:

```bash
docker compose up -d qdrant
```

Qdrant HTTP runs at:

```text
http://localhost:6333
```

Default config:

```yaml
qdrant:
  url: http://localhost:6333
  collection: face_embeddings
  vector_size: 512
  top_k: 5
  max_registration_images: 5
```

Collection settings:

```text
collection: face_embeddings
vector size: 512
distance: cosine
```

Registration replaces all previous vectors for the same employee ID and stores up to 5 face embeddings. Verification queries the top 5 closest vectors, chooses the employee ID that appears most often, and uses the highest score from that winning ID as confidence.

Register up to 5 images using repeated `file` fields:

```bash
curl -X POST http://127.0.0.1:8000/api/verification/register \
  -F "person_id=employee_001" \
  -F "person_name=Employee One" \
  -F "file=@/path/to/face1.jpg" \
  -F "file=@/path/to/face2.jpg" \
  -F "file=@/path/to/face3.jpg"
```

Verify:

```bash
curl -X POST http://127.0.0.1:8000/api/verification/verify \
  -F "file=@/path/to/face.jpg"
```

Local ONNX Runtime mode loads models from:

```text
backend/weights/<model_name>.onnx
```

Current expected files:

```text
backend/weights/
  face_detection.onnx
  anti_spoofing.onnx
  emotion.onnx
  resnet18_face.onnx
```

Run local ONNX mode:

```bash
BACKEND_USE_TRITON=0 uvicorn app.main:app --reload
```

Run Triton mode:

```bash
BACKEND_USE_TRITON=1 TRITON_URL=localhost:8000 uvicorn app.main:app --reload
```

Each service should implement:

```python
preprocess(image: np.ndarray) -> dict[str, np.ndarray]
postprocess(outputs: dict[str, np.ndarray]) -> Any
```

`BaseService.predict()` handles loading the backend, running inference, and passing raw outputs into `postprocess()`.

## Service Contracts

### Face Detection

File:

```text
app/services/face_detection_service.py
```

Public method:

```python
detect(image: PIL.Image.Image) -> list[dict]
```

Required output format:

```python
[
    {
        "bbox": (x, y, w, h),
        "confidence": 0.95,
        "crop": face_crop_image,
        "keypoints": [
            (0.36, 0.31),
            (0.44, 0.31),
            (0.40, 0.38),
            (0.37, 0.46),
            (0.43, 0.46),
        ],
    }
]
```

Rules:

- `bbox` must be normalized floats in `[0, 1]`.
- `bbox` format is `(x, y, w, h)`, not `(x1, y1, x2, y2)`.
- `keypoints` must contain 5 normalized `(x, y)` points in this order: left eye, right eye, nose tip, left mouth corner, right mouth corner.
- `crop` must be a `PIL.Image.Image`.
- Return an empty list when no faces are found.

API response shape:

```json
{
  "image_width": 1280,
  "image_height": 720,
  "faces": [
    {
      "bbox": {"x": 0.3, "y": 0.2, "w": 0.2, "h": 0.3},
      "detection_confidence": 0.95,
      "crop_width": 224,
      "crop_height": 224,
      "keypoints": [
        {"x": 0.36, "y": 0.31},
        {"x": 0.44, "y": 0.31},
        {"x": 0.40, "y": 0.38},
        {"x": 0.37, "y": 0.46},
        {"x": 0.43, "y": 0.46}
      ]
    }
  ]
}
```

### Anti-Spoofing / Liveness

File:

```text
app/services/anti_spoofing_service.py
```

Public method:

```python
predict(face_images: PIL.Image.Image | list[PIL.Image.Image]) -> list[dict]
```

Required output format:

```python
[
    {
        "label": "real",
        "confidence": 0.92,
    }
]
```

Rules:

- Always return a list, even if the input is one image.
- Valid labels for the service layer are `"real"` or `"fake"`.
- `confidence` must be a float from `0.0` to `1.0`.
- The pipeline treats a face as live when label is `"real"` and confidence is above the configured threshold.

API response shape:

```json
{
  "anti_spoofing": {
    "label": "real",
    "confidence": 0.92
  }
}
```

### Emotion

File:

```text
app/services/emotion_service.py
```

Public method:

```python
predict(face_images: PIL.Image.Image | list[PIL.Image.Image]) -> list[dict]
```

Required output format:

```python
[
    {
        "label": "happy",
        "confidence": 0.88,
    }
]
```

Rules:

- Always return a list, even if the input is one image.
- `label` is a string such as `"neutral"`, `"happy"`, `"sad"`, `"angry"`, `"surprise"`, `"fear"`, or `"disgust"`.
- `confidence` must be a float from `0.0` to `1.0`.
- Output order must match input face order.

API response shape:

```json
{
  "emotion": {
    "label": "happy",
    "confidence": 0.88
  }
}
```

### Verification / Recognition

File:

```text
app/services/verification_service.py
```

Current local model:

```text
backend/weights/resnet18_face.onnx
```

Public methods:

```python
register(face_images: PIL.Image.Image | list[PIL.Image.Image], person_id: str) -> dict
verify(face_images: PIL.Image.Image | list[PIL.Image.Image]) -> list[dict]
```

`register()` required output:

```python
{
    "person_id": "employee_001",
    "status": "registered",
    "image_count": 5,
}
```

Valid `status` values:

```text
registered
updated
```

`verify()` required output:

```python
[
    {
        "employee_id": "employee_001",
        "employee_name": "employee_001",
        "label": "employee_001",
        "confidence": 0.82,
        "matched": True,
    }
]
```

Rules:

- Always return a list, even if the input is one image.
- Verification supports batched ResNet18 inference and processes face crops in chunks of 32.
- Registration stores 1 to 5 embeddings per employee in Qdrant.
- Model outputs are stored raw; this backend does not L2-normalize them before Qdrant.
- Verification searches the top 5 Qdrant matches, picks the employee ID that appears most often, and reports the highest score for that winning ID.
- Output order must match input face order.
- `confidence` is cosine similarity.
- `matched` must be `True` only when similarity is above the match threshold.
- For unknown faces, use:

```python
{
    "employee_id": None,
    "employee_name": "unknown",
    "label": "unknown",
    "confidence": 0.0,
    "matched": False,
}
```

API verify response shape:

```json
{
  "recognition": {
    "label": "employee_001",
    "confidence": 0.82,
    "matched": true
  }
}
```

API register response shape:

```json
{
  "person_id": "employee_001",
  "status": "registered",
  "message": "Stored 5 face embedding(s) in Qdrant."
}
```

## Pipeline Contract

File:

```text
app/services/inference_service.py
```

Pipeline order:

```text
1. Face detection on the full image
2. Anti-spoofing on each face crop
3. Emotion on live face crops only
4. Verification on live face crops only
5. Merge results back into one response
```

Pipeline internal face result:

```python
FaceResult(
    bbox=(x, y, w, h),
    detection_score=0.95,
    keypoints=[(x, y), (x, y), (x, y), (x, y), (x, y)],
    is_live=True,
    liveness_score=0.92,
    emotion="happy",
    emotion_score=0.88,
    employee_id="employee_001",
    employee_name="employee_001",
    similarity=0.82,
    verified=True,
)
```

Pipeline API response shape:

```json
{
  "image_width": 1280,
  "image_height": 720,
  "faces": [
    {
      "face": {
        "bbox": {"x": 0.3, "y": 0.2, "w": 0.2, "h": 0.3},
        "detection_confidence": 0.95,
        "crop_width": 0,
        "crop_height": 0,
        "keypoints": [
          {"x": 0.36, "y": 0.31},
          {"x": 0.44, "y": 0.31},
          {"x": 0.40, "y": 0.38},
          {"x": 0.37, "y": 0.46},
          {"x": 0.43, "y": 0.46}
        ]
      },
      "emotion": {
        "label": "happy",
        "confidence": 0.88
      },
      "anti_spoofing": {
        "label": "real",
        "confidence": 0.92
      },
      "recognition": {
        "label": "employee_001",
        "confidence": 0.82,
        "matched": true
      }
    }
  ]
}
```

## Testing

Start the backend:

```bash
cd backend
BACKEND_USE_TRITON=0 uvicorn app.main:app --reload
```

Test the full frame pipeline:

```bash
python test_frame.py --image /path/to/image.jpg
```

Test verification registration:

```bash
curl -X POST http://127.0.0.1:8000/api/verification/register \
  -F "person_id=employee_001" \
  -F "file=@/path/to/face1.jpg"
```

Test verification:

```bash
curl -X POST http://127.0.0.1:8000/api/verification/verify \
  -F "file=@/path/to/face2.jpg"
```

## Developer Checklist

Before opening a pull request, make sure:

- The service returns the exact contract documented above.
- Batch methods return one output per input image, in the same order.
- Model files are placed in `backend/weights/`.
- Local model names match the service model name.
- No route returns raw NumPy arrays or PIL images.
- Confidence values are plain Python floats, not NumPy scalar types.
- Empty detections return `[]`, not `None`.
- Unknown recognition results use `"unknown"` and `matched=False`.
