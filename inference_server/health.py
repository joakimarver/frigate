"""FastAPI health endpoint for the remote inference server.

Runs in a background daemon thread so it does not block the ZMQ server loop.

Endpoints
---------
GET /health
    Returns server status, loaded models, request counters, and mean
    inference latency.

    Response (200 OK, application/json)::

        {
          "status": "ok",
          "uptime_s": 123.4,
          "requests_total": 5000,
          "inference_total": 4980,
          "mean_latency_ms": 8.31,
          "loaded_models": ["model.onnx"]
        }

GET /health/ready
    Returns 200 if at least one model is loaded, 503 otherwise.
    Suitable for Kubernetes readinessProbe / Docker HEALTHCHECK.

GET /health/live
    Always returns 200 while the process is running.
    Suitable for Kubernetes livenessProbe.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference_server.server import ServerHealth

logger = logging.getLogger(__name__)


def start_health_server(health: ServerHealth, port: int) -> None:
    """Start the FastAPI health server in a background daemon thread.

    Args:
        health: A :class:`~inference_server.server.ServerHealth` instance.
        port:   TCP port to bind on (e.g. 5556).
    """
    try:
        import uvicorn
        from fastapi import FastAPI, Response
    except ImportError:
        logger.warning(
            "fastapi/uvicorn not installed — health endpoint disabled.  "
            "Install with: pip install fastapi uvicorn[standard]"
        )
        return

    app = FastAPI(title="Frigate Inference Server", docs_url=None, redoc_url=None)

    @app.get("/health")
    def get_health() -> dict:
        data = health.snapshot()
        data["status"] = "ok"
        return data

    @app.get("/health/ready")
    def get_ready(response: Response) -> dict:
        data = health.snapshot()
        if not data["loaded_models"]:
            response.status_code = 503
            return {"status": "not_ready", "reason": "no models loaded"}
        return {"status": "ready", "loaded_models": data["loaded_models"]}

    @app.get("/health/live")
    def get_live() -> dict:
        return {"status": "alive"}

    cfg = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",  # keep uvicorn quiet; inference_server logs separately
        access_log=False,
    )
    server = uvicorn.Server(cfg)

    thread = threading.Thread(
        target=server.run,
        name="health-server",
        daemon=True,
    )
    thread.start()
