"""Entry point for the Frigate remote inference server.

Usage::

    python -m inference_server [options]

Or via the installed console script::

    frigate-inference-server [options]
"""

from __future__ import annotations

import argparse
import logging

from inference_server.config import ServerConfig
from inference_server.server import run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="frigate-inference-server",
        description=(
            "Frigate remote GPU inference server.  "
            "Receives detection requests from Frigate over ZMQ and returns "
            "results using ONNX Runtime (CUDA / DirectML / ROCm / CPU)."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=(
            "ZMQ bind address.  Default: tcp://*:5555 (or $INFERENCE_ENDPOINT env var)"
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        dest="model_dir",
        help=(
            "Directory to store received model files.  "
            "Default: ~/.frigate-inference/models (or $INFERENCE_MODEL_DIR)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Inference device: auto | cuda | directml | rocm | cpu.  "
            "Default: auto (or $INFERENCE_DEVICE)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.  Default: INFO",
    )
    parser.add_argument(
        "--health-port",
        default=None,
        dest="health_port",
        type=int,
        help=(
            "Port for the HTTP health endpoint (GET /health).  "
            "Set to 0 to disable.  Default: 5556 (or $INFERENCE_HEALTH_PORT)"
        ),
    )
    parser.add_argument(
        "--workers",
        default=None,
        type=int,
        help=(
            "Number of concurrent inference worker threads.  "
            "Default: 4 (or $INFERENCE_WORKERS)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = ServerConfig(
        endpoint=args.endpoint,
        model_dir=args.model_dir,
        device=args.device,
        log_level=args.log_level,
        health_port=args.health_port,
        workers=args.workers,
    )

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run(config)


if __name__ == "__main__":
    main()
