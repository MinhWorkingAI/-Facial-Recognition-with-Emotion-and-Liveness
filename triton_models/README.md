# Triton Model Repository

This folder contains Triton model configs for the backend services.

The ONNX files stay in `backend/weights` and are mounted into the Triton
container at runtime:

- `backend/weights/face_detection.onnx`
- `backend/weights/anti_spoofing.onnx`
- `backend/weights/emotion.onnx`
- `backend/weights/resnet18_face.onnx`

`start-triton.sh` creates Triton's required layout inside the container:

```text
/tmp/triton_models/
  face_detection/1/model.onnx
  anti_spoofing/1/model.onnx
  emotion/1/model.onnx
  resnet18_face/1/model.onnx
```
