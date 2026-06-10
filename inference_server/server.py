"""Concurrent ZMQ ROUTER/DEALER inference server.

Architecture
============
The original single-threaded REQ/REP design processed one frame at a time,
causing every camera to queue behind the slowest frame.  This module replaces
it with a **ROUTER/DEALER** pattern backed by a thread pool so that multiple
inference requests can execute concurrently.

                      ┌─────────────────────────────────────┐
  Frigate clients ──► │  ROUTER  (frontend, tcp//*:5555)    │
                      │      ↕  zmq.proxy via inproc pair   │
                      │  DEALER  (backend, inproc)           │
                      └──────────┬──────────────────────────┘
                                 │  (one REP socket per worker)
                        ┌────────┴────────────────────────────┐
                        │  Worker thread  Worker thread  …    │
                        │  (ThreadPoolExecutor, N workers)    │
                        └─────────────────────────────────────┘

Each worker thread owns a single ZMQ REP socket connected to the DEALER
backend.  The proxy forwards full multipart envelopes (address frames + data
frames), so workers get the complete client message and respond normally.

Protocol (unchanged from original REP server)
=============================================
Inference request (multipart):
    frame 0: JSON  {"shape": [...], "dtype": "...", "model_type": "..."}
    frame 1: raw tensor bytes

Model check request (single frame):
    {"model_request": true, "model_name": "<name>"}

Model upload request (multipart):
    frame 0: JSON  {"model_data": true, "model_name": "<name>"}
    frame 1: model file bytes

Hot-swap request (multipart) — item 7:
    frame 0: JSON  {"model_reload": true, "model_name": "<name>"}
    frame 1: new model file bytes
    Response: JSON {"reloaded": bool, "model_name": "<name>"}

Responses:
    Inference     → [JSON header, raw float32 bytes (480 B)]
    Model check   → [JSON {"model_available": bool, "model_loaded": bool}]
    Model upload  → [JSON {"model_saved": bool, "model_loaded": bool}]
    Hot-swap      → [JSON {"reloaded": bool, "model_name": "<name>"}]
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zmq

from inference_server.config import ServerConfig
from inference_server.model_store import ModelStore

logger = logging.getLogger(__name__)

_ZERO_RESULT = np.zeros((20, 6), np.float32)

# Internal inproc address used to connect the proxy DEALER → worker REP sockets.
_BACKEND_ADDR = "inproc://inference-workers"


# ---------------------------------------------------------------------------
# Shared health state
# ---------------------------------------------------------------------------


class ServerHealth:
    """Thread-safe container for server health statistics.

    Consumed by the FastAPI health endpoint (health.py).
    """

    def __init__(self, store: ModelStore) -> None:
        self._lock = threading.Lock()
        self._store = store
        self._start_time: float = time.time()
        self._requests_total: int = 0
        self._inference_total: int = 0
        # Running sum and count for computing mean latency (ms)
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0

    def record_request(self) -> None:
        with self._lock:
            self._requests_total += 1

    def record_inference(self, latency_ms: float) -> None:
        with self._lock:
            self._inference_total += 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            mean_latency = (
                self._latency_sum_ms / self._latency_count
                if self._latency_count
                else 0.0
            )
            return {
                "uptime_s": round(time.time() - self._start_time, 1),
                "requests_total": self._requests_total,
                "inference_total": self._inference_total,
                "mean_latency_ms": round(mean_latency, 2),
                "loaded_models": self._store.loaded_model_names(),
            }


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


def _encode_detections(detections: np.ndarray) -> list[bytes]:
    """Encode a (20, 6) float32 array into a two-frame ZMQ reply."""
    header = json.dumps({"shape": list(detections.shape), "dtype": "float32"}).encode()
    return [header, detections.astype(np.float32).tobytes()]


# ---------------------------------------------------------------------------
# Request handlers  (stateless functions, safe to call from multiple threads)
# ---------------------------------------------------------------------------


def _handle_model_check(header: dict, store: ModelStore) -> list[bytes]:
    model_name: str = header.get("model_name", "")
    available = store.is_available(model_name)
    loaded = store.is_loaded(model_name)

    if available and not loaded:
        loaded = store.load_from_disk(model_name)

    return [json.dumps({"model_available": available, "model_loaded": loaded}).encode()]


def _handle_model_upload(
    header: dict, model_bytes: bytes, store: ModelStore
) -> list[bytes]:
    model_name: str = header.get("model_name", "")
    ok = store.save_and_load(model_name, model_bytes)
    return [json.dumps({"model_saved": ok, "model_loaded": ok}).encode()]


def _handle_model_reload(
    header: dict, model_bytes: bytes, store: ModelStore
) -> list[bytes]:
    """Atomic hot-swap: overwrite the on-disk file and replace the runner.

    The store's ``save_and_load`` creates a new ``OnnxRunner`` before assigning
    it to ``_runners[name]``, so in-flight requests using the old runner will
    complete normally.  The dict update is atomic from Python's GIL perspective.
    """
    model_name: str = header.get("model_name", "")
    ok = store.save_and_load(model_name, model_bytes)
    if ok:
        logger.info("Hot-swapped model %r", model_name)
    else:
        logger.error("Hot-swap failed for model %r", model_name)
    return [json.dumps({"reloaded": ok, "model_name": model_name}).encode()]


def _handle_inference(
    header: dict,
    tensor_bytes: bytes,
    store: ModelStore,
    health: ServerHealth,
) -> list[bytes]:
    model_type: str = header.get("model_type", "ssd")
    shape = tuple(header.get("shape", []))
    dtype = np.dtype(header.get("dtype", "float32"))

    model_name: str = header.get("model_name", "")
    if not model_name:
        loaded = store.loaded_model_names()
        if not loaded:
            logger.warning("No model loaded — returning zeros")
            return _encode_detections(_ZERO_RESULT)
        model_name = loaded[0]

    try:
        tensor = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
        t0 = time.monotonic()
        detections = store.run(model_name, tensor, model_type)
        latency_ms = (time.monotonic() - t0) * 1000.0
        health.record_inference(latency_ms)
    except Exception as exc:
        logger.error("Inference error for model %r: %s", model_name, exc)
        detections = _ZERO_RESULT

    return _encode_detections(detections)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


def _worker(
    ctx: zmq.Context,
    store: ModelStore,
    health: ServerHealth,
    stop_event: threading.Event,
) -> None:
    """Single worker thread: owns one REP socket connected to the DEALER."""
    sock: zmq.Socket = ctx.socket(zmq.REP)
    sock.connect(_BACKEND_ADDR)

    while not stop_event.is_set():
        try:
            if not sock.poll(timeout=300):
                continue
            frames = sock.recv_multipart()
        except zmq.ZMQError as exc:
            if stop_event.is_set():
                break
            logger.error("Worker ZMQ error: %s", exc)
            continue

        health.record_request()

        if not frames:
            sock.send_multipart([b"{}"])
            continue

        try:
            header: dict = json.loads(frames[0].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Worker: bad header: %s", exc)
            sock.send_multipart([json.dumps({"error": "bad header"}).encode()])
            continue

        try:
            if "model_request" in header:
                reply = _handle_model_check(header, store)
            elif "model_reload" in header:
                model_bytes = frames[1] if len(frames) > 1 else b""
                reply = _handle_model_reload(header, model_bytes, store)
            elif "model_data" in header:
                model_bytes = frames[1] if len(frames) > 1 else b""
                reply = _handle_model_upload(header, model_bytes, store)
            else:
                tensor_bytes = frames[1] if len(frames) > 1 else b""
                reply = _handle_inference(header, tensor_bytes, store, health)
        except Exception as exc:
            logger.exception("Worker: unhandled error: %s", exc)
            reply = [json.dumps({"error": "internal error"}).encode()]

        try:
            sock.send_multipart(reply)
        except zmq.ZMQError as exc:
            logger.error("Worker: send error: %s", exc)

    sock.close(linger=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(config: ServerConfig) -> None:
    """Start the concurrent ROUTER/DEALER server and block until shutdown."""
    store = ModelStore(config.model_dir, config.device)
    store.preload_all()

    health = ServerHealth(store)

    context = zmq.Context()

    # Frontend: Frigate clients connect here.
    frontend: zmq.Socket = context.socket(zmq.ROUTER)
    frontend.bind(config.endpoint)

    # Backend: workers connect here via inproc.
    backend: zmq.Socket = context.socket(zmq.DEALER)
    backend.bind(_BACKEND_ADDR)

    logger.info(
        "Inference server listening on %s (%d workers)",
        config.endpoint,
        config.workers,
    )
    logger.info("Model directory: %s", config.model_dir)
    logger.info("Device: %s", config.device)

    # ---- health endpoint ---------------------------------------------------
    if config.health_port > 0:
        from inference_server.health import start_health_server

        start_health_server(health, config.health_port)
        logger.info("Health endpoint at http://0.0.0.0:%d/health", config.health_port)

    # ---- graceful shutdown -------------------------------------------------
    stop_event = threading.Event()

    def _on_signal(signum, frame):  # noqa: ANN001
        logger.info("Received signal %s — shutting down", signum)
        stop_event.set()
        # Close the frontend socket to interrupt zmq.proxy()
        try:
            frontend.close(linger=0)
        except zmq.ZMQError:
            pass

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    # ------------------------------------------------------------------------

    # Start worker threads.
    with ThreadPoolExecutor(
        max_workers=config.workers, thread_name_prefix="inf-worker"
    ) as pool:
        for _ in range(config.workers):
            pool.submit(_worker, context, store, health, stop_event)

        # Run the ZMQ proxy on the main thread — it blocks until the frontend
        # socket is closed (which happens in _on_signal above).
        try:
            zmq.proxy(frontend, backend)
        except zmq.ZMQError:
            # proxy raises ZMQError when the frontend socket is closed; that is
            # the expected shutdown path.
            pass

        stop_event.set()

    try:
        backend.close(linger=0)
    except zmq.ZMQError:
        pass
    context.term()
    logger.info("Inference server stopped")
    sys.exit(0)
