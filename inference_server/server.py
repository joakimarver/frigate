"""ZMQ REP server — receives frames from Frigate and returns detections.

Protocol (mirrors ``frigate/detectors/plugins/zmq_ipc.py``):

Inference request (multipart):
    frame 0: JSON  {"shape": [...], "dtype": "...", "model_type": "..."}
    frame 1: raw tensor bytes

Model check request (single frame):
    {"model_request": true, "model_name": "<name>"}

Model upload request (multipart):
    frame 0: JSON  {"model_data": true, "model_name": "<name>"}
    frame 1: model file bytes

Responses:
    Inference     → [JSON header, raw float32 bytes (480 B)]
    Model check   → [JSON {"model_available": bool, "model_loaded": bool}]
    Model upload  → [JSON {"model_saved": bool, "model_loaded": bool}]
"""

from __future__ import annotations

import json
import logging
import signal
import sys

import numpy as np
import zmq

from inference_server.config import ServerConfig
from inference_server.model_store import ModelStore

logger = logging.getLogger(__name__)

_ZERO_RESULT = np.zeros((20, 6), np.float32)


def _encode_detections(detections: np.ndarray) -> list[bytes]:
    """Encode a (20, 6) float32 array into a two-frame ZMQ reply."""
    header = json.dumps({"shape": list(detections.shape), "dtype": "float32"}).encode()
    return [header, detections.astype(np.float32).tobytes()]


def _handle_model_check(
    header: dict,
    store: ModelStore,
    socket: zmq.Socket,
) -> None:
    model_name: str = header.get("model_name", "")
    available = store.is_available(model_name)
    loaded = store.is_loaded(model_name)

    # If it's on disk but not yet loaded, try loading now.
    if available and not loaded:
        loaded = store.load_from_disk(model_name)

    reply = json.dumps({"model_available": available, "model_loaded": loaded}).encode()
    socket.send_multipart([reply])


def _handle_model_upload(
    header: dict,
    model_bytes: bytes,
    store: ModelStore,
    socket: zmq.Socket,
) -> None:
    model_name: str = header.get("model_name", "")
    ok = store.save_and_load(model_name, model_bytes)
    reply = json.dumps({"model_saved": ok, "model_loaded": ok}).encode()
    socket.send_multipart([reply])


def _handle_inference(
    header: dict,
    tensor_bytes: bytes,
    store: ModelStore,
    socket: zmq.Socket,
) -> None:
    model_type: str = header.get("model_type", "ssd")
    shape = tuple(header.get("shape", []))
    dtype = np.dtype(header.get("dtype", "float32"))

    # Determine which model to use (use the first loaded model if none specified).
    model_name: str = header.get("model_name", "")
    if not model_name:
        loaded = [n for n in store._runners]  # noqa: SLF001 — intentional access
        if not loaded:
            logger.warning("No model loaded — returning zeros")
            socket.send_multipart(_encode_detections(_ZERO_RESULT))
            return
        model_name = loaded[0]

    try:
        tensor = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
        detections = store.run(model_name, tensor, model_type)
    except Exception as exc:
        logger.error("Inference error for model %r: %s", model_name, exc)
        detections = _ZERO_RESULT

    socket.send_multipart(_encode_detections(detections))


def run(config: ServerConfig) -> None:
    """Start the ZMQ REP server and block until a SIGINT/SIGTERM is received."""
    store = ModelStore(config.model_dir, config.device)
    store.preload_all()

    context = zmq.Context()
    socket: zmq.Socket = context.socket(zmq.REP)
    socket.bind(config.endpoint)
    logger.info("Inference server listening on %s", config.endpoint)
    logger.info("Model directory: %s", config.model_dir)
    logger.info("Device: %s", config.device)

    # ---- graceful shutdown on SIGINT / SIGTERM ---------------------------
    _stop = False

    def _on_signal(signum, frame):  # noqa: ANN001
        nonlocal _stop
        logger.info("Received signal %s — shutting down", signum)
        _stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    # -----------------------------------------------------------------------

    while not _stop:
        try:
            # Poll with a 500 ms timeout so we can check _stop regularly.
            if not socket.poll(timeout=500):
                continue

            frames = socket.recv_multipart()
        except zmq.ZMQError as exc:
            if _stop:
                break
            logger.error("ZMQ receive error: %s", exc)
            continue

        if not frames:
            socket.send_multipart([b"{}"])
            continue

        try:
            header: dict = json.loads(frames[0].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Failed to parse request header: %s", exc)
            socket.send_multipart([json.dumps({"error": "bad header"}).encode()])
            continue

        try:
            if "model_request" in header:
                _handle_model_check(header, store, socket)
            elif "model_data" in header:
                model_bytes = frames[1] if len(frames) > 1 else b""
                _handle_model_upload(header, model_bytes, store, socket)
            else:
                tensor_bytes = frames[1] if len(frames) > 1 else b""
                _handle_inference(header, tensor_bytes, store, socket)
        except Exception as exc:
            logger.exception("Unhandled error processing request: %s", exc)
            # Must always send a reply to keep REQ/REP state machine intact.
            try:
                socket.send_multipart(
                    [json.dumps({"error": "internal error"}).encode()]
                )
            except zmq.ZMQError:
                pass

    socket.close(linger=0)
    context.term()
    logger.info("Inference server stopped")
    sys.exit(0)
