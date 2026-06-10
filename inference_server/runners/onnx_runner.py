"""ONNX-based inference runner for the remote inference server.

Supports CUDA, DirectML (Windows), ROCm (Linux AMD), and CPU execution
providers, selected automatically based on what is available at runtime.

The ``OnnxRunner`` produces a ``(20, 6)`` float32 array matching
Frigate's internal detection format so results can be returned directly
over the ZMQ protocol without any further transformation in the server.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import onnxruntime as ort

from inference_server.runners.post_process import (
    _CONF_THRESHOLD,
    post_process_dfine,
    post_process_rfdetr,
    post_process_yolo,
    post_process_yolox,
)

logger = logging.getLogger(__name__)

# Supported model type string constants (must match Frigate's ModelTypeEnum values).
_MODEL_TYPES = {
    "dfine",
    "rfdetr",
    "ssd",
    "yolox",
    "yolonas",
    "yolo-generic",
}

# YOLOX grid/stride strides — must match Frigate's DetectionApi.calculate_grids_strides
_YOLOX_STRIDES = [8, 16, 32]


def _get_providers(device: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Return ORT execution-provider list and options for *device*.

    device values (case-insensitive): auto, cuda, directml, rocm, cpu
    """
    device = device.lower()
    available = ort.get_available_providers()
    logger.debug("Available ORT providers: %s", available)

    if device == "cpu":
        return ["CPUExecutionProvider"], [{}]

    candidates: list[tuple[str, dict[str, Any]]] = []

    # Explicit device selection
    if device == "cuda" and "CUDAExecutionProvider" in available:
        candidates.append(("CUDAExecutionProvider", {"device_id": 0}))
    elif device == "directml" and "DmlExecutionProvider" in available:
        candidates.append(("DmlExecutionProvider", {"device_id": 0}))
    elif device in ("rocm", "migraphx") and "MIGraphXExecutionProvider" in available:
        candidates.append(("MIGraphXExecutionProvider", {}))
    elif device == "auto":
        # Priority: CUDA > DirectML > ROCm/MIGraphX > CPU
        if "CUDAExecutionProvider" in available:
            candidates.append(("CUDAExecutionProvider", {"device_id": 0}))
        if "DmlExecutionProvider" in available:
            candidates.append(("DmlExecutionProvider", {"device_id": 0}))
        if "MIGraphXExecutionProvider" in available:
            candidates.append(("MIGraphXExecutionProvider", {}))

    # Always add CPU as the final fallback
    candidates.append(("CPUExecutionProvider", {}))

    providers = [p for p, _ in candidates]
    options = [o for _, o in candidates]
    logger.info("Selected ORT providers: %s", providers)
    return providers, options


def _calculate_grids_strides(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute YOLOX grids and expanded strides (mirrors Frigate's helper)."""
    grids = []
    expanded_strides = []

    hsizes = [height // stride for stride in _YOLOX_STRIDES]
    wsizes = [width // stride for stride in _YOLOX_STRIDES]

    for hsize, wsize, stride in zip(hsizes, wsizes, _YOLOX_STRIDES):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    return np.concatenate(grids, 1), np.concatenate(expanded_strides, 1)


class OnnxRunner:
    """Loads an ONNX model and runs inference, returning (20, 6) detections."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        providers, options = _get_providers(device)
        self._session = ort.InferenceSession(
            model_path,
            providers=providers,
            provider_options=options,
        )
        self._input_name: str = self._session.get_inputs()[0].name
        # Determine model dimensions from the first input shape.
        # Shape is typically (1, H, W, 3) or (1, 3, H, W).
        shape = self._session.get_inputs()[0].shape
        if shape[1] == 3:
            # NCHW
            self._height = int(shape[2])
            self._width = int(shape[3])
        else:
            # NHWC
            self._height = int(shape[1])
            self._width = int(shape[2])

        # Pre-compute YOLOX grids (only used for yolox model type)
        self._yolox_grids, self._yolox_strides = _calculate_grids_strides(
            self._height, self._width
        )

        logger.info(
            "OnnxRunner: loaded %s  input=%s  shape=%s  providers=%s",
            model_path,
            self._input_name,
            shape,
            self._session.get_providers(),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, tensor_input: np.ndarray, model_type: str) -> np.ndarray:
        """Run inference and return a (20, 6) float32 detection array.

        Args:
            tensor_input: Pre-processed frame tensor (as sent by Frigate's
                          ZMQ client) with shape (1, H, W, C) and the dtype
                          that the model expects.
            model_type:   One of the model type strings defined in
                          ``ModelTypeEnum`` (e.g. ``"yolox"``, ``"ssd"``).

        Returns:
            numpy array of shape (20, 6) in Frigate detection format.
        """
        model_type = model_type.lower()

        if model_type == "dfine":
            output = self._session.run(
                None,
                {
                    self._input_name: tensor_input,
                    "orig_target_sizes": np.array(
                        [[self._height, self._width]], dtype=np.int64
                    ),
                },
            )
            return post_process_dfine(output, self._width, self._height)

        output = self._session.run(None, {self._input_name: tensor_input})

        if model_type == "rfdetr":
            return post_process_rfdetr(output)

        if model_type == "yolonas":
            predictions = output[0]
            detections = np.zeros((20, 6), np.float32)
            for i, prediction in enumerate(predictions):
                if i == 20:
                    break
                (_, x_min, y_min, x_max, y_max, confidence, class_id) = prediction
                if class_id < 0:
                    break
                detections[i] = [
                    class_id,
                    confidence,
                    y_min / self._height,
                    x_min / self._width,
                    y_max / self._height,
                    x_max / self._width,
                ]
            return detections

        if model_type == "yolo-generic":
            return post_process_yolo(output, self._width, self._height)

        if model_type == "yolox":
            return post_process_yolox(
                output[0],
                self._width,
                self._height,
                self._yolox_grids.copy(),
                self._yolox_strides.copy(),
            )

        if model_type == "ssd":
            # SSD output is already in Frigate's (20, 6) format or
            # needs to be assembled from boxes/classes/scores outputs.
            # Most SSD models return [boxes, classes, scores, num_detections].
            if len(output) >= 3:
                return _post_process_ssd(output, self._width, self._height)
            # If we get a single (20,6) array, return directly
            return np.asarray(output[0], dtype=np.float32).reshape(20, 6)

        logger.warning("Unknown model_type %r — returning zeros", model_type)
        return np.zeros((20, 6), np.float32)


# ---------------------------------------------------------------------------
# SSD post-processing (for completeness — covers TF-Lite style SSD models
# that expose separate boxes / classes / scores outputs)
# ---------------------------------------------------------------------------


def _post_process_ssd(
    output: list[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """Handle SSD-style models with separate boxes/classes/scores outputs."""
    # Typical output order for TFLite SSD: boxes, classes, scores, num_detections
    boxes = np.squeeze(output[0])  # (N, 4) [y_min, x_min, y_max, x_max] normalized
    classes = np.squeeze(output[1])  # (N,)
    scores = np.squeeze(output[2])  # (N,)

    detections = np.zeros((20, 6), np.float32)
    det_idx = 0
    for i in range(min(len(scores), 20)):
        if scores[i] < _CONF_THRESHOLD:
            break
        detections[det_idx] = [
            classes[i],
            scores[i],
            boxes[i][0],
            boxes[i][1],
            boxes[i][2],
            boxes[i][3],
        ]
        det_idx += 1

    return detections
