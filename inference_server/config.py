"""Configuration for the Frigate remote inference server."""

import os


class ServerConfig:
    """Configuration loaded from CLI arguments or environment variables."""

    def __init__(
        self,
        endpoint: str | None = None,
        model_dir: str | None = None,
        device: str | None = None,
        log_level: str | None = None,
    ) -> None:
        self.endpoint: str = endpoint or os.environ.get(
            "INFERENCE_ENDPOINT", "tcp://*:5555"
        )
        self.model_dir: str = model_dir or os.environ.get(
            "INFERENCE_MODEL_DIR",
            os.path.join(os.path.expanduser("~"), ".frigate-inference", "models"),
        )
        self.device: str = (
            device or os.environ.get("INFERENCE_DEVICE", "auto")
        ).lower()
        self.log_level: str = (
            log_level or os.environ.get("INFERENCE_LOG_LEVEL", "info")
        ).upper()
