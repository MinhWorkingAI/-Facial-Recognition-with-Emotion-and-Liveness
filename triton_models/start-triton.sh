#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPOSITORY="${SOURCE_REPOSITORY:-/model-repository}"
RUNTIME_REPOSITORY="${RUNTIME_REPOSITORY:-/tmp/triton_models}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/weights}"

MODELS=(
  "face_detection:face_detection.onnx"
  "anti_spoofing:anti_spoofing.onnx"
  "emotion:emotion.onnx"
  "resnet18_face:resnet18_face.onnx"
)

rm -rf "${RUNTIME_REPOSITORY}"
mkdir -p "${RUNTIME_REPOSITORY}"

for model_entry in "${MODELS[@]}"; do
  model_name="${model_entry%%:*}"
  weight_file="${model_entry##*:}"
  weight_path="${WEIGHTS_DIR}/${weight_file}"

  if [[ ! -f "${weight_path}" ]]; then
    echo "Missing ONNX weight for Triton model '${model_name}': ${weight_path}" >&2
    exit 1
  fi

  mkdir -p "${RUNTIME_REPOSITORY}/${model_name}/1"
  cp "${SOURCE_REPOSITORY}/${model_name}/config.pbtxt" \
    "${RUNTIME_REPOSITORY}/${model_name}/config.pbtxt"
  ln -s "${weight_path}" "${RUNTIME_REPOSITORY}/${model_name}/1/model.onnx"
done

exec tritonserver \
  --model-repository="${RUNTIME_REPOSITORY}" \
  --strict-model-config=true
