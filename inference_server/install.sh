#!/usr/bin/env bash
# install.sh — Install the Frigate remote inference server on Linux
#
# Usage:
#   ./install.sh [--device cpu|cuda|rocm|auto]
#
# Requirements:
#   - Python 3.10+
#   - pip
#   - For CUDA: CUDA Toolkit 11.x or 12.x and matching NVIDIA drivers
#   - For ROCm: ROCm 5.x+ and a supported AMD GPU
#
# The script creates a virtual environment at ./venv and installs all
# dependencies. After installation you can start the server with:
#   ./venv/bin/python -m inference_server [options]
# or:
#   ./start.sh [options]

set -euo pipefail

DEVICE="auto"

# --- Parse arguments -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        -h|--help)
            grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "=== Frigate Remote Inference Server — Linux Installer ==="
echo "Device: $DEVICE"
echo "Install directory: $SCRIPT_DIR"
echo ""

# --- Check Python -----------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# --- Create virtual environment ---------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

"$PIP" install --upgrade pip --quiet

# --- Install core dependencies ----------------------------------------------
echo "Installing core dependencies..."
"$PIP" install --quiet pyzmq numpy "opencv-python-headless>=4.8"

# --- Install ONNX Runtime variant -------------------------------------------
case "$DEVICE" in
    cuda)
        echo "Installing onnxruntime-gpu (CUDA)..."
        "$PIP" install --quiet "onnxruntime-gpu>=1.17"
        ;;
    rocm)
        echo "Installing onnxruntime-gpu (ROCm)..."
        # ROCm variant is distributed as onnxruntime-gpu with a ROCm index
        "$PIP" install --quiet "onnxruntime-gpu>=1.17" \
            --extra-index-url https://download.pytorch.org/whl/rocm6.0 2>/dev/null || \
        "$PIP" install --quiet "onnxruntime-gpu>=1.17"
        ;;
    cpu)
        echo "Installing onnxruntime (CPU only)..."
        "$PIP" install --quiet "onnxruntime>=1.17"
        ;;
    auto)
        # Auto-detect: try CUDA first, then plain onnxruntime
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
            echo "NVIDIA GPU detected — installing onnxruntime-gpu (CUDA)..."
            "$PIP" install --quiet "onnxruntime-gpu>=1.17" || {
                echo "onnxruntime-gpu install failed; falling back to CPU..."
                "$PIP" install --quiet "onnxruntime>=1.17"
            }
        elif command -v rocminfo &>/dev/null && rocminfo &>/dev/null 2>&1; then
            echo "AMD GPU (ROCm) detected — installing onnxruntime-gpu (ROCm)..."
            "$PIP" install --quiet "onnxruntime-gpu>=1.17" || {
                echo "onnxruntime-gpu install failed; falling back to CPU..."
                "$PIP" install --quiet "onnxruntime>=1.17"
            }
        else
            echo "No GPU detected — installing onnxruntime (CPU)..."
            "$PIP" install --quiet "onnxruntime>=1.17"
        fi
        ;;
    *)
        echo "ERROR: Unknown device '$DEVICE'. Valid values: auto, cuda, rocm, cpu"
        exit 1
        ;;
esac

# --- Create start script ----------------------------------------------------
cat > "$SCRIPT_DIR/start.sh" << 'STARTSCRIPT'
#!/usr/bin/env bash
# start.sh — Start the Frigate remote inference server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/venv/bin/python" -m inference_server "$@"
STARTSCRIPT
chmod +x "$SCRIPT_DIR/start.sh"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Start the server:"
echo "  cd $SCRIPT_DIR && ./start.sh"
echo ""
echo "Common options:"
echo "  ./start.sh --endpoint tcp://*:5555"
echo "  ./start.sh --device cuda"
echo "  ./start.sh --model-dir /path/to/models"
echo "  ./start.sh --log-level DEBUG"
echo ""
echo "See README.md for full configuration details."
