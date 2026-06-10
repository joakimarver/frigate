# Frigate Remote Inference Server

Run GPU-accelerated object detection on a **separate machine** connected via LAN, with automatic fallback to the local CPU/GPU if the remote server is unavailable.

## Overview

Frigate uses ZeroMQ to send frames to detectors. The remote inference server implements the same ZMQ protocol, so Frigate treats it like any other detector. The `fallback` detector plugin adds transparent failover: Frigate uses the remote GPU first, and automatically switches to the local detector if the network server goes down.

```
┌──────────────────────────────┐      ZMQ TCP      ┌─────────────────────────────────┐
│  Frigate  (NVR machine)      │ ─────────────────► │  Inference Server  (GPU machine)│
│                              │    LAN (port 5555) │                                 │
│  zmq detector plugin         │ ◄───────────────── │  ONNX Runtime (CUDA/DirectML)   │
│  fallback detector plugin    │   detections        │  Windows or Linux               │
└──────────────────────────────┘                    └─────────────────────────────────┘
```

## Prerequisites

### GPU machine (inference server)

- Python 3.10 or later
- For **NVIDIA CUDA** (Linux or Windows): CUDA Toolkit 11.x or 12.x + matching drivers
- For **AMD/Intel GPU on Windows**: Windows 10 1903+ (DirectML is built-in, no extra drivers needed)
- For **AMD ROCm** (Linux): ROCm 5.x+
- For **CPU only**: nothing extra

### Frigate machine

- Frigate NVR running normally
- Network access to the GPU machine on port 5555 (or your chosen port)

---

## Quick Start

### 1. Set up the inference server (GPU machine)

**Linux:**

```bash
# Clone or copy the inference_server directory to your GPU machine
cd inference_server

# Run the installer (auto-detects GPU)
chmod +x install.sh
./install.sh

# Start the server
./start.sh
```

**Windows (PowerShell or Command Prompt):**

```bat
cd inference_server
install.bat
start.bat
```

The server listens on `tcp://*:5555` by default and logs to stdout.

### 2. Configure Frigate

Edit `config/config.yml` on the Frigate machine:

```yaml
detectors:
  # Remote GPU server
  remote_gpu:
    type: zmq
    endpoint: "tcp://192.168.1.50:5555"   # replace with your GPU machine's IP
    request_timeout_ms: 1000

  # Local fallback (used automatically if remote is unreachable)
  local_cpu:
    type: onnx
    device: CPU

  # Fallback detector — uses remote_gpu first, falls back to local_cpu
  detection:
    type: fallback
    primary: remote_gpu
    secondary: local_cpu
    health_check_interval_s: 30   # how often to retry the primary
```

Restart Frigate. It will now send frames to the GPU server and fall back to local CPU automatically.

---

## Installation Options

### Linux installer

```bash
./install.sh [--device auto|cuda|rocm|cpu]
```

| Flag | Description |
|------|-------------|
| `--device auto` | Auto-detect GPU (default) |
| `--device cuda` | Force NVIDIA CUDA |
| `--device rocm` | Force AMD ROCm |
| `--device cpu` | CPU only, no GPU |

### Windows installer

```bat
install.bat [--device auto|cuda|directml|cpu]
```

| Flag | Description |
|------|-------------|
| `--device auto` | Auto-detect (default; tries CUDA then DirectML) |
| `--device cuda` | Force NVIDIA CUDA |
| `--device directml` | AMD/Intel/NVIDIA via DirectML |
| `--device cpu` | CPU only |

---

## Server Options

```
python -m inference_server [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--endpoint` | `tcp://*:5555` | ZMQ bind address |
| `--model-dir` | `~/.frigate-inference/models` | Directory to cache received models |
| `--device` | `auto` | `auto`, `cuda`, `directml`, `rocm`, `cpu` |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--health-port` | `5556` | HTTP health endpoint port; set to `0` to disable |
| `--workers` | `4` | Number of concurrent inference worker threads |

All options can also be set via environment variables:

| Environment variable | Corresponding flag |
|---------------------|-------------------|
| `INFERENCE_ENDPOINT` | `--endpoint` |
| `INFERENCE_MODEL_DIR` | `--model-dir` |
| `INFERENCE_DEVICE` | `--device` |
| `INFERENCE_LOG_LEVEL` | `--log-level` |
| `INFERENCE_HEALTH_PORT` | `--health-port` |
| `INFERENCE_WORKERS` | `--workers` |

---

## Firewall

Open port **5555/TCP** (or your chosen port) on the GPU machine:

**Linux (ufw):**
```bash
sudo ufw allow 5555/tcp
```

**Linux (firewalld):**
```bash
sudo firewall-cmd --add-port=5555/tcp --permanent && sudo firewall-cmd --reload
```

**Windows (PowerShell, run as Administrator):**
```powershell
New-NetFirewallRule -DisplayName "Frigate Inference" -Direction Inbound -Protocol TCP -LocalPort 5555 -Action Allow
```

> **Security note:** ZMQ TCP has no built-in authentication. Only expose port 5555 on a trusted LAN. For internet-facing setups, use a VPN or SSH tunnel.

---

## Running as a Service

### Linux (systemd)

Create `/etc/systemd/system/frigate-inference.service`:

```ini
[Unit]
Description=Frigate Remote Inference Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/inference_server
ExecStart=/path/to/inference_server/venv/bin/python -m inference_server
Restart=on-failure
RestartSec=5
Environment=INFERENCE_ENDPOINT=tcp://*:5555
Environment=INFERENCE_DEVICE=auto

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now frigate-inference
sudo systemctl status frigate-inference
```

### Windows (Task Scheduler)

1. Open **Task Scheduler** → **Create Basic Task**
2. Trigger: **At system startup**
3. Action: **Start a program** → `C:\path\to\inference_server\start.bat`
4. Check **Run whether user is logged on or not**

### Windows (Service — recommended)

The `install_service.bat` script installs the server as a proper Windows service that starts automatically at boot and restarts on failure.

```bat
# Run as Administrator
install_service.bat

# Custom options
install_service.bat --name FrigateInference --endpoint "tcp://*:5555" --device cuda --workers 8

# Remove the service
install_service.bat --remove
```

For best results, install [NSSM](https://nssm.cc/download) (Non-Sucking Service Manager) and add it to `PATH` before running the script. NSSM provides automatic restart on failure, stdout/stderr log files with rotation, and more reliable service lifecycle management than the `sc.exe` fallback.

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | `FrigateInference` | Windows service name |
| `--endpoint` | `tcp://*:5555` | ZMQ bind address |
| `--device` | `auto` | `auto`, `cuda`, `directml`, `cpu` |
| `--health-port` | `5556` | HTTP health port (`0` = disabled) |
| `--workers` | `4` | Inference worker threads |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--remove` | — | Uninstall the service |

---

## Concurrent Inference

The server uses a ZMQ **ROUTER/DEALER** architecture backed by a thread pool, so multiple cameras can submit inference requests simultaneously without queuing behind each other.

```
Frigate cameras ──► ROUTER (tcp/:5555) ──► DEALER (inproc)
                                                │
                              ┌─────────────────┼──────────────────┐
                              │                 │                  │
                         Worker 1          Worker 2  …         Worker N
                        (REP socket)      (REP socket)        (REP socket)
                       ONNX Runtime      ONNX Runtime        ONNX Runtime
```

Each worker thread owns its own ZMQ REP socket. Adjust the number of workers with `--workers` (default: 4). For GPU inference, the ONNX Runtime handles its own internal parallelism, so 4–8 workers is usually sufficient. For CPU-only setups, match `--workers` to your core count.

---

## Health Endpoint

When `--health-port` is set to a non-zero value (default: 5556), the server starts an HTTP server with three endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Full status: uptime, request counts, mean latency, loaded models |
| `GET /health/ready` | `200` if at least one model is loaded; `503` otherwise |
| `GET /health/live` | Always `200` while the process is running |

Example response from `GET /health`:

```json
{
  "status": "ok",
  "uptime_s": 123.4,
  "requests_total": 5000,
  "inference_total": 4980,
  "mean_latency_ms": 8.31,
  "loaded_models": ["my_model.onnx"]
}
```

The health endpoint requires `fastapi` and `uvicorn` to be installed. The server starts normally without them, but health checks will be unavailable.

```bash
pip install fastapi "uvicorn[standard]"
```

Disable the health endpoint:

```bash
python -m inference_server --health-port 0
```

Also open port 5556 in your firewall if you want to monitor the server from another machine:

```bash
sudo ufw allow 5556/tcp
```

---

## Hot-Swap Models

You can replace a loaded model without restarting the server. Send a `model_reload` ZMQ message from a Frigate client or custom script:

```python
import zmq, json

ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://192.168.1.50:5555")

with open("new_model.onnx", "rb") as f:
    model_bytes = f.read()

sock.send_multipart([
    json.dumps({"model_reload": True, "model_name": "new_model.onnx"}).encode(),
    model_bytes,
])
reply = json.loads(sock.recv_multipart()[0])
print(reply)  # {"reloaded": true, "model_name": "new_model.onnx"}
```

In-flight inference requests using the old model complete normally before the new model takes over.

---

## Fallback Detector Behavior

The `fallback` plugin in Frigate handles automatic switching:

1. **Normal operation**: All frames are sent to the primary (remote GPU).
2. **Primary failure**: If the primary raises an exception (e.g. a ZMQ timeout when the remote server is unreachable), Frigate switches to the secondary (local CPU/GPU) and logs an error.
3. **Recovery**: Every `health_check_interval_s` seconds (default: 30), Frigate tries the primary again. If it succeeds, it stays on primary; if it fails, it immediately returns to secondary.

> **Note:** The fallback triggers on *exceptions* only, not on all-zero detection results. An all-zero result is the normal response when no objects are present in the frame — treating it as a failure would cause constant primary/secondary switching on cameras monitoring empty scenes.

```yaml
detectors:
  detection:
    type: fallback
    primary: remote_gpu        # name of the preferred detector
    secondary: local_cpu       # name of the fallback detector
    health_check_interval_s: 30
```

Set `health_check_interval_s: 0` to disable automatic recovery (stay on secondary until Frigate restarts).

---

## Supported Models

The inference server supports all ONNX models that Frigate supports:

| Model type | Config value |
|-----------|-------------|
| SSD | `ssd` |
| YOLOX | `yolox` |
| YOLO-NAS | `yolonas` |
| YOLO (generic) | `yolo-generic` |
| D-FINE | `dfine` |
| RT-DETR (RF-DETR) | `rfdetr` |

Set the model type in your Frigate config:

```yaml
model:
  path: /config/model_cache/my_model.onnx
  model_type: yolox
  width: 640
  height: 640
  input_tensor: nchw
  input_pixel_format: rgb
```

When Frigate connects to the ZMQ server, it automatically transfers the model file if the server does not already have it cached.

---

## Troubleshooting

### Server does not start

- Check Python version: `python3 --version` (need 3.10+)
- Check ONNX Runtime is installed: `python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
- Run with `--log-level DEBUG` for more detail

### Frigate keeps using the fallback

- Verify the server is running: `curl` or `nc -zv <IP> 5555`
- Check the GPU machine's firewall (see Firewall section above)
- Increase `request_timeout_ms` in the `zmq` detector config (try 2000–5000 for slow networks)
- Check Frigate logs for ZMQ timeout messages

### CUDA not detected

- Verify CUDA is installed: `nvidia-smi`
- Verify the ONNX Runtime CUDA provider is available:
  ```bash
  python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
  ```
  Should include `CUDAExecutionProvider`.
- Reinstall with explicit device: `./install.sh --device cuda`

### DirectML not detected (Windows)

- Requires Windows 10 1903 or later
- Check that `DmlExecutionProvider` appears in ORT providers:
  ```bat
  venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_available_providers())"
  ```
- Reinstall with: `install.bat --device directml`

---

## Directory Structure

```
inference_server/
├── __init__.py             # Package marker
├── __main__.py             # CLI entry point (python -m inference_server)
├── config.py               # ServerConfig (CLI args + env vars)
├── server.py               # ROUTER/DEALER concurrent ZMQ server
├── health.py               # FastAPI HTTP health endpoint
├── model_store.py          # Disk-based model cache + runner management
├── runners/
│   ├── __init__.py
│   ├── onnx_runner.py      # ONNX Runtime with GPU/CPU EP auto-selection
│   └── post_process.py     # Post-processing (YOLOX, DFINE, RFDETR, etc.)
├── requirements.txt        # Dependency reference
├── install.sh              # Linux installer
├── install.bat             # Windows installer
├── install_service.bat     # Windows service installer (run as Administrator)
└── README.md               # This file

frigate/detectors/plugins/
└── fallback.py             # Fallback/priority detector plugin (Frigate-side)

frigate/config/config.py
└── (modified)              # Wires fallback detector references at config validation
```

---

## What to Implement Next

The following improvements are planned for future iterations:

1. **TLS / CURVE authentication** — add ZMQ CURVE so only authorized Frigate instances can connect; prevents untrusted clients from submitting arbitrary tensors.

2. ~~**Multi-model concurrency**~~ ✅ — ROUTER/DEALER with `--workers` thread pool (see [Concurrent Inference](#concurrent-inference)).

3. ~~**Health endpoint**~~ ✅ — FastAPI on `--health-port` with `/health`, `/health/ready`, `/health/live` (see [Health Endpoint](#health-endpoint)).

4. **Metrics / Prometheus** — emit `prometheus_client` metrics (inference latency histogram, frames/sec, model load times) for Grafana dashboards.

5. ~~**Windows service installer**~~ ✅ — `install_service.bat` with NSSM support (see [Windows Service](#windows-service--recommended)).

6. **Docker image** — provide a `Dockerfile.inference-server` so the server can run as a container alongside Nvidia Container Toolkit or ROCm Docker on Linux.

7. ~~**Dynamic model hot-swap**~~ ✅ — zero-downtime model reload via `model_reload` ZMQ message (see [Hot-Swap Models](#hot-swap-models)).

8. **Batch inference** — buffer multiple camera frames and run them as a single batch through the GPU for better throughput on multi-camera systems.

9. **ROCm / HIP runner** — add an explicit `rocm_runner.py` that uses the MIGraphX execution provider and handles ROCm-specific quirks (currently handled via generic ORT auto-detection).

10. **Model quantization helper** — add a utility script that takes an FP32 ONNX model and converts it to INT8/FP16 using ORT quantization tools, providing a further speed boost for GPU inference.
