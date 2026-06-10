"""Fallback detector plugin — tries a primary detector and falls back to a
secondary one if the primary is unavailable or returns all-zero results.

This allows a remote GPU inference server (via the ``zmq`` detector) to be
configured as the primary detector with a local CPU/GPU detector as the
backup.  Frigate will transparently switch between them without any manual
intervention.

Example ``config.yml``::

    detectors:
      remote_gpu:
        type: zmq
        endpoint: "tcp://192.168.1.50:5555"
        request_timeout_ms: 1000
      local_cpu:
        type: onnx
        device: CPU
      detection:
        type: fallback
        primary: remote_gpu
        secondary: local_cpu

.. note::
   The ``fallback`` detector shares the same ``DetectorRunner`` process as
   every other detector.  The ``primary`` and ``secondary`` fields must name
   two **other** detectors defined in the same ``detectors:`` block.  Those
   referenced detectors are instantiated *inside* the fallback; they do not
   get their own separate processes.
"""

import logging
import time

import numpy as np
from pydantic import ConfigDict, Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)

DETECTOR_KEY = "fallback"


class FallbackDetectorConfig(BaseDetectorConfig):
    """Wraps two detectors and falls back from *primary* to *secondary* on failure.

    The ``primary`` and ``secondary`` fields must reference the **names** of
    other detectors configured in ``config.yml`` (not their types).
    """

    model_config = ConfigDict(title="Fallback")

    type: Literal[DETECTOR_KEY]

    primary: str = Field(
        title="Primary detector name",
        description=(
            "Name of the preferred detector (e.g. a remote ZMQ GPU detector). "
            "Must match a key in the top-level ``detectors`` config."
        ),
    )
    secondary: str = Field(
        title="Secondary (fallback) detector name",
        description=(
            "Name of the fallback detector used when the primary is unavailable. "
            "Must match a key in the top-level ``detectors`` config."
        ),
    )
    health_check_interval_s: float = Field(
        default=30.0,
        title="Health-check interval (seconds)",
        description=(
            "How often (in seconds) the fallback detector attempts to switch back "
            "to the primary after a failure.  Set to 0 to disable automatic recovery."
        ),
    )


def _instantiate_detector(
    config: BaseDetectorConfig | None,
    name: str,
) -> "DetectionApi | None":
    """Instantiate a detector from its config, returning None on failure."""
    if config is None:
        logger.error("FallbackDetector: no config found for detector %r", name)
        return None

    from frigate.detectors.detector_types import api_types

    cls = api_types.get(config.type)
    if cls is None:
        logger.error(
            "FallbackDetector: unknown detector type %r for %r", config.type, name
        )
        return None

    try:
        instance = cls(config)
        logger.info("FallbackDetector: instantiated %r (type=%s)", name, config.type)
        return instance
    except Exception as exc:
        logger.error("FallbackDetector: failed to instantiate %r: %s", name, exc)
        return None


class FallbackDetector(DetectionApi):
    """Detector that delegates to *primary* with automatic fallback to *secondary*."""

    type_key = DETECTOR_KEY

    def __init__(self, detector_config: FallbackDetectorConfig) -> None:
        super().__init__(detector_config)

        self._config = detector_config
        self._using_primary: bool = True
        self._last_health_check: float = 0.0
        self._zero_result = np.zeros((20, 6), np.float32)

        # The Frigate config validator injects the referenced detector configs
        # as private attributes _primary_config / _secondary_config before
        # the FallbackDetector is instantiated.
        primary_cfg: BaseDetectorConfig | None = getattr(
            detector_config, "_primary_config", None
        )
        secondary_cfg: BaseDetectorConfig | None = getattr(
            detector_config, "_secondary_config", None
        )

        self._primary = _instantiate_detector(primary_cfg, detector_config.primary)
        self._secondary = _instantiate_detector(
            secondary_cfg, detector_config.secondary
        )

        if self._primary is None and self._secondary is None:
            logger.error(
                "FallbackDetector: neither primary nor secondary could be instantiated"
            )
        elif self._primary is None:
            logger.warning(
                "FallbackDetector: primary %r unavailable; using secondary only",
                detector_config.primary,
            )
            self._using_primary = False

    # ------------------------------------------------------------------
    # DetectionApi interface
    # ------------------------------------------------------------------

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        # Periodic attempt to recover the primary.
        self._maybe_try_primary_recovery()

        active = self._primary if self._using_primary else self._secondary
        if active is None:
            logger.warning("FallbackDetector: no active detector — returning zeros")
            return self._zero_result

        try:
            result = active.detect_raw(tensor_input)
            # A fully-zero result from the primary (e.g. ZMQ timeout) is treated
            # as a failure signal so we can switch to secondary promptly.
            if self._using_primary and np.all(result == 0):
                logger.warning(
                    "FallbackDetector: primary %r returned all-zero detections — "
                    "switching to secondary %r",
                    self._config.primary,
                    self._config.secondary,
                )
                self._using_primary = False
                self._last_health_check = time.monotonic()
                return self._run_secondary(tensor_input)
            return result
        except Exception as exc:
            if self._using_primary:
                logger.error(
                    "FallbackDetector: primary %r raised an exception (%s) — "
                    "switching to secondary %r",
                    self._config.primary,
                    exc,
                    self._config.secondary,
                )
                self._using_primary = False
                self._last_health_check = time.monotonic()
            return self._run_secondary(tensor_input)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_try_primary_recovery(self) -> None:
        """Periodically switch back to the primary detector to check if it recovered."""
        interval = self._config.health_check_interval_s
        if self._using_primary or interval <= 0 or self._primary is None:
            return

        now = time.monotonic()
        if now - self._last_health_check >= interval:
            logger.info(
                "FallbackDetector: attempting recovery to primary %r",
                self._config.primary,
            )
            self._using_primary = True
            self._last_health_check = now

    def _run_secondary(self, tensor_input: np.ndarray) -> np.ndarray:
        if self._secondary is None:
            return self._zero_result
        try:
            return self._secondary.detect_raw(tensor_input)
        except Exception as exc:
            logger.error(
                "FallbackDetector: secondary %r also raised an exception: %s",
                self._config.secondary,
                exc,
            )
            return self._zero_result
