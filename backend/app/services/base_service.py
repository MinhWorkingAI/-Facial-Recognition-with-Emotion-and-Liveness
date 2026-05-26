"""
base_service.py
===============
Abstract base class for all ML inference services.

Each concrete service (liveness, emotion, face detection, …) must:
  1. Inherit from ``BaseService``.
  2. Pass the model name used by Triton and by local ONNX weights.
  3. Implement ``preprocess``, ``postprocess``, and optionally ``predict``.

Inference can run through Triton HTTP or a local ONNX Runtime GPU session.
All image handling is done with numpy arrays (uint8 RGB, shape H×W×3).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import tritonclient.http as triton_http
from tritonclient.utils import InferenceServerException

from app.config import settings

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """
    Base class for services that call either Triton or a local ONNX model.

    Parameters
    ----------
    model_name : str
        Name of the model as registered in Triton and used to find local weights.
    use_triton : bool | None
        ``True`` uses Triton. ``False`` loads ``.onnx`` from ``weights_dir``.
        ``None`` reads ``BACKEND_USE_TRITON`` from app settings.
    triton_url : str
        Triton HTTP endpoint, e.g. ``"localhost:8000"``.
    model_version : str
        Model version string.  ``""`` means "latest".
    request_timeout : int
        Per-request timeout in seconds for Triton HTTP calls.
    weights_dir : str | Path | None
        Root folder for local ONNX weights. Defaults to ``backend/weights``.
        Local models are loaded as ``<weights_dir>/<model_name>.onnx``.
    model_path : str | Path | None
        Explicit local ONNX model path. Overrides ``weights_dir`` resolution.
    providers : list[str] | None
        ONNX Runtime providers. Defaults to CUDA first, then CPU fallback.
    """

    def __init__(
        self,
        model_name: str,
        use_triton: bool | None = None,
        triton_url: str | None = None,
        model_version: str | None = None,
        request_timeout: int | None = None,
        weights_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self.use_triton = settings.use_triton if use_triton is None else use_triton
        self.triton_url = triton_url or settings.triton_url
        self.model_name = model_name
        self.model_version = settings.model_version if model_version is None else model_version
        self.request_timeout = int(
            settings.request_timeout if request_timeout is None else request_timeout
        )
        self.weights_dir = Path(weights_dir or settings.weights_dir).expanduser()
        self.model_path = Path(model_path).expanduser() if model_path is not None else None
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._client: triton_http.InferenceServerClient | None = None
        self._onnx_session: Any | None = None
        self._input_metadata: list[dict] | None = None
        self._output_metadata: list[dict] | None = None

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Load the configured inference backend and fetch model metadata.
        Call once at application startup (e.g. FastAPI ``lifespan``).
        """
        if self.use_triton:
            self._connect_triton()
        else:
            self._connect_onnx()

    def _connect_triton(self) -> None:
        logger.info(
            "[%s] Connecting to Triton at %s (model=%s, version=%s)",
            self.__class__.__name__,
            self.triton_url,
            self.model_name,
            self.model_version or "latest",
        )
        self._client = triton_http.InferenceServerClient(
            url=self.triton_url,
            verbose=False,
        )

        if not self._client.is_server_live():
            raise RuntimeError(
                f"Triton server at {self.triton_url!r} is not live."
            )
        if not self._client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(
                f"Model '{self.model_name}' (version='{self.model_version}') "
                f"is not ready on Triton at {self.triton_url!r}."
            )

        metadata = self._client.get_model_metadata(
            self.model_name, self.model_version
        )
        self._input_metadata = metadata["inputs"]
        self._output_metadata = metadata["outputs"]

        logger.info(
            "[%s] Connected. inputs=%s  outputs=%s",
            self.__class__.__name__,
            [m["name"] for m in self._input_metadata],
            [m["name"] for m in self._output_metadata],
        )

    def _connect_onnx(self) -> None:
        model_path = self._resolve_model_path()
        logger.info(
            "[%s] Loading ONNX model at %s (providers=%s)",
            self.__class__.__name__,
            model_path,
            self.providers,
        )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime-gpu is required for local model inference. "
                "Install it with `pip install onnxruntime-gpu`."
            ) from exc

        available_providers = ort.get_available_providers()
        selected_providers = [
            provider for provider in self.providers if provider in available_providers
        ]
        if not selected_providers:
            raise RuntimeError(
                "None of the requested ONNX Runtime providers are available. "
                f"requested={self.providers}, available={available_providers}"
            )
        if "CUDAExecutionProvider" not in selected_providers:
            logger.warning(
                "[%s] CUDAExecutionProvider is not available; local inference will use %s.",
                self.__class__.__name__,
                selected_providers,
            )

        self._onnx_session = ort.InferenceSession(
            str(model_path),
            providers=selected_providers,
        )
        self._input_metadata = [
            {
                "name": input_meta.name,
                "datatype": input_meta.type,
                "shape": input_meta.shape,
            }
            for input_meta in self._onnx_session.get_inputs()
        ]
        self._output_metadata = [
            {
                "name": output_meta.name,
                "datatype": output_meta.type,
                "shape": output_meta.shape,
            }
            for output_meta in self._onnx_session.get_outputs()
        ]

        logger.info(
            "[%s] ONNX model loaded. inputs=%s  outputs=%s",
            self.__class__.__name__,
            [m["name"] for m in self._input_metadata],
            [m["name"] for m in self._output_metadata],
        )

    def _resolve_model_path(self) -> Path:
        if self.model_path is not None:
            if self.model_path.is_file():
                return self.model_path
            raise FileNotFoundError(f"ONNX model file not found: {self.model_path}")

        model_path = self.weights_dir / f"{self.model_name}.onnx"
        if model_path.is_file():
            return model_path

        raise FileNotFoundError(
            f"Could not find local ONNX model for '{self.model_name}'. "
            f"Expected: {model_path}"
        )

    def close(self) -> None:
        """Close the loaded backend. Call at application shutdown."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("[%s] Triton client closed.", self.__class__.__name__)
        self._onnx_session = None

    @property
    def client(self) -> triton_http.InferenceServerClient:
        if self._client is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.connect() must be called before inference."
            )
        return self._client

    @property
    def onnx_session(self) -> Any:
        if self._onnx_session is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.connect() must be called before inference."
            )
        return self._onnx_session

    # ------------------------------------------------------------------
    # Inference pipeline
    # ------------------------------------------------------------------

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convert a raw image into model input tensors.

        Parameters
        ----------
        image : np.ndarray
            Input image, uint8 RGB, shape ``(H, W, 3)``.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of model input name → numpy array ready for inference.
        """

    @abstractmethod
    def postprocess(self, outputs: dict[str, np.ndarray]) -> Any:
        """
        Convert raw model output tensors into a usable result.

        Parameters
        ----------
        outputs : dict[str, np.ndarray]
            Mapping of model output name → numpy array.

        Returns
        -------
        Any
            Domain-specific result (label, embedding, dict, …).
        """

    def predict(self, image: np.ndarray) -> Any:
        """
        End-to-end inference: preprocess → model backend → postprocess.

        Parameters
        ----------
        image : np.ndarray
            Input image, uint8 RGB, shape ``(H, W, 3)``.

        Returns
        -------
        Any
            Result of ``postprocess``.

        Raises
        ------
        InferenceServerException
            Propagated from tritonclient on server-side errors.
        ValueError
            If the image array is not valid (wrong dtype, wrong ndim).
        """
        self._validate_image(image)
        self._ensure_loaded()

        input_tensors = self.preprocess(image)
        raw_outputs = self._infer(input_tensors)

        return self.postprocess(raw_outputs)

    def _ensure_loaded(self) -> None:
        if self.use_triton:
            if self._client is None:
                self.connect()
        elif self._onnx_session is None:
            self.connect()

    def _infer(self, input_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._input_metadata is None or self._output_metadata is None:
            raise RuntimeError(f"{self.__class__.__name__}.connect() did not load metadata.")

        if self.use_triton:
            return self._infer_triton(input_tensors)
        return self._infer_onnx(input_tensors)

    def _infer_triton(self, input_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._input_metadata is None or self._output_metadata is None:
            raise RuntimeError(f"{self.__class__.__name__}.connect() did not load metadata.")

        infer_inputs = []
        for meta in self._input_metadata:
            name = meta["name"]
            if name not in input_tensors:
                raise KeyError(
                    f"preprocess() did not return tensor '{name}' "
                    f"required by model '{self.model_name}'."
                )
            arr = input_tensors[name]
            infer_input = triton_http.InferInput(name, arr.shape, meta["datatype"])
            infer_input.set_data_from_numpy(arr)
            infer_inputs.append(infer_input)

        # Request only the outputs declared in metadata
        infer_outputs = [
            triton_http.InferRequestedOutput(m["name"])
            for m in self._output_metadata
        ]

        # Triton HTTP call
        try:
            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=infer_inputs,
                outputs=infer_outputs,
                timeout=self.request_timeout,
            )
        except InferenceServerException as exc:
            logger.error(
                "[%s] Triton inference failed: %s", self.__class__.__name__, exc
            )
            raise

        return {
            m["name"]: response.as_numpy(m["name"])
            for m in self._output_metadata
        }

    def _infer_onnx(self, input_tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._input_metadata is None or self._output_metadata is None:
            raise RuntimeError(f"{self.__class__.__name__}.connect() did not load metadata.")

        feed: dict[str, np.ndarray] = {}
        for meta in self._input_metadata:
            name = meta["name"]
            if name not in input_tensors:
                raise KeyError(
                    f"preprocess() did not return tensor '{name}' "
                    f"required by model '{self.model_name}'."
                )
            feed[name] = input_tensors[name]

        output_names = [m["name"] for m in self._output_metadata]
        output_values = self.onnx_session.run(output_names, feed)
        return dict(zip(output_names, output_values))

    def _validate_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy.ndarray.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape (H, W, 3).")
        if image.dtype != np.uint8:
            raise ValueError("image must have dtype uint8.")
