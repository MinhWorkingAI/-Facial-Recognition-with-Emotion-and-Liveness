"""
Export the trained FASD MobileNetV2 Keras model (.keras or .h5) to ONNX.

The saved model already includes MobileNetV2 preprocessing and the classifier
head. Feed uint8/float RGB images in [0, 255] with shape (N, 224, 224, 3).

Example:
    python export_onnx.py
    python export_onnx.py model/fasd_mobilenetv2_model.h5 model/fasd_mobilenetv2_model.onnx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import onnx
import tensorflow as tf
import tf2onnx

MODULE_DIR = Path(__file__).resolve().parent
MODEL_DIR = MODULE_DIR / "model"
DEFAULT_MODEL_STEM = "fasd_mobilenetv2_model"
IMG_HEIGHT = 224
IMG_WIDTH = 224
DEFAULT_OPSET = 13
SUPPORTED_SUFFIXES = {".keras", ".h5"}


def resolve_model_path(model_path: Path | None) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported model format '{path.suffix}'. "
                f"Use one of: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
            )
        return path

    for suffix in (".keras", ".h5"):
        candidate = MODEL_DIR / f"{DEFAULT_MODEL_STEM}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No model found. Expected {DEFAULT_MODEL_STEM}.keras or "
        f"{DEFAULT_MODEL_STEM}.h5 under {MODEL_DIR}"
    )


def default_onnx_path(model_path: Path) -> Path:
    return model_path.with_suffix(".onnx")


def load_keras_model(model_path: Path) -> tf.keras.Model:
    suffix = model_path.suffix.lower()

    if suffix == ".keras":
        print(f"Loading Keras model from: {model_path}")
        return tf.keras.models.load_model(str(model_path), compile=False)

    if suffix == ".h5":
        try:
            print(f"Loading Keras model from: {model_path}")
            return tf.keras.models.load_model(str(model_path), compile=False)
        except (ValueError, OSError) as exc:
            keras_sibling = model_path.with_suffix(".keras")
            if keras_sibling.exists():
                print(
                    f"Warning: {model_path.name} uses legacy TensorFlow op layers "
                    f"that Keras 3 cannot deserialize ({exc}). "
                    f"Loading equivalent weights from {keras_sibling.name}."
                )
                return tf.keras.models.load_model(
                    str(keras_sibling), compile=False
                )
            raise ValueError(
                f"Unable to load {model_path.name}. This HDF5 export embeds "
                "TensorFlow op layers (TrueDivide/Subtract from MobileNetV2 "
                "preprocessing) that Keras 3 no longer deserializes. Re-save "
                "with model.save('model.keras') or place a matching .keras "
                "file alongside the .h5 weights."
            ) from exc

    raise ValueError(
        f"Unsupported model format '{suffix}'. "
        f"Use one of: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def export_onnx(
    model_path: Path,
    onnx_path: Path,
    *,
    opset: int = DEFAULT_OPSET,
    verify: bool = True,
) -> Path:
    model = load_keras_model(model_path)

    input_signature = (
        tf.TensorSpec(
            [None, IMG_HEIGHT, IMG_WIDTH, 3],
            tf.float32,
            name="input",
        ),
    )

    @tf.function(input_signature=input_signature)
    def serving_fn(x: tf.Tensor) -> tf.Tensor:
        return model(x, training=False)

    print(f"Converting to ONNX (opset {opset})...")
    model_proto, _ = tf2onnx.convert.from_function(
        serving_fn,
        input_signature=input_signature,
        opset=opset,
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, str(onnx_path))
    print(f"Saved ONNX model to: {onnx_path}")

    if verify:
        _verify_export(model, onnx_path)

    return onnx_path


def _verify_export(model: tf.keras.Model, onnx_path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("Skipping verification: onnxruntime is not installed.")
        return

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    keras_out = model(dummy, training=False).numpy()
    onnx_out = session.run([output_name], {input_name: dummy})[0]

    max_diff = float(np.max(np.abs(keras_out - onnx_out)))
    print(
        "Verification OK "
        f"(input={input_name!r}, output={output_name!r}, max_diff={max_diff:.3e})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FASD MobileNetV2 .keras/.h5 weights to ONNX."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to .keras or .h5 model (default: model/fasd_mobilenetv2_model.*)",
    )
    parser.add_argument(
        "onnx_path",
        nargs="?",
        help="Output .onnx path (default: same name as input, .onnx extension)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help=f"ONNX opset version (default: {DEFAULT_OPSET})",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip numeric parity check against the Keras model",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    model_path = resolve_model_path(
        Path(args.model_path) if args.model_path else None
    )
    onnx_path = (
        Path(args.onnx_path)
        if args.onnx_path
        else default_onnx_path(model_path)
    )

    export_onnx(
        model_path,
        onnx_path,
        opset=args.opset,
        verify=not args.no_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
