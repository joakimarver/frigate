"""Disk-based model cache for the remote inference server."""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class ModelStore:
    """Manages saving, loading, and caching of ONNX model files.

    Models are stored on disk under ``model_dir`` and loaded into an
    ONNX-based runner on demand.  The store is also responsible for
    creating and returning a :class:`~inference_server.runners.onnx_runner.OnnxRunner`
    for each model.
    """

    def __init__(self, model_dir: str, device: str) -> None:
        self._model_dir = model_dir
        self._device = device
        # name -> OnnxRunner instance
        self._runners: dict[str, object] = {}
        os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def model_path(self, model_name: str) -> str:
        return os.path.join(self._model_dir, model_name)

    def is_available(self, model_name: str) -> bool:
        return os.path.isfile(self.model_path(model_name))

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._runners

    def save_and_load(self, model_name: str, model_bytes: bytes) -> bool:
        """Persist *model_bytes* to disk and load it into a runner.

        Returns True on success, False on any failure.
        """
        path = self.model_path(model_name)
        try:
            with open(path, "wb") as fh:
                fh.write(model_bytes)
            logger.info("Saved model %s (%d bytes)", model_name, len(model_bytes))
        except OSError as exc:
            logger.error("Failed to save model %s: %s", model_name, exc)
            return False

        return self._load(model_name, path)

    def load_from_disk(self, model_name: str) -> bool:
        """Load a previously saved model from disk.

        Returns True on success, False if the file does not exist or loading fails.
        """
        path = self.model_path(model_name)
        if not os.path.isfile(path):
            return False
        return self._load(model_name, path)

    def run(
        self, model_name: str, tensor_input: np.ndarray, model_type: str
    ) -> np.ndarray:
        """Run inference and return a (20, 6) float32 detection array."""
        runner = self._runners.get(model_name)
        if runner is None:
            raise RuntimeError(f"Model {model_name!r} is not loaded")
        return runner.run(tensor_input, model_type)

    def preload_all(self) -> None:
        """Load every model file already present in ``model_dir``."""
        for fname in os.listdir(self._model_dir):
            if fname.endswith(".onnx"):
                self.load_from_disk(fname)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, model_name: str, path: str) -> bool:
        # Import here so that the store module doesn't hard-depend on
        # onnxruntime at import time (simplifies unit testing).
        from inference_server.runners.onnx_runner import OnnxRunner

        try:
            runner = OnnxRunner(path, self._device)
            self._runners[model_name] = runner
            logger.info("Loaded model %s on device=%s", model_name, self._device)
            return True
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            return False
