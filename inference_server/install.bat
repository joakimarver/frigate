@echo off
REM install.bat — Install the Frigate remote inference server on Windows
REM
REM Usage:
REM   install.bat [--device cpu^|cuda^|directml^|auto]
REM
REM Requirements:
REM   - Python 3.10+ (from python.org or Microsoft Store)
REM   - pip
REM   - For CUDA: CUDA Toolkit 11.x or 12.x and NVIDIA drivers
REM   - For DirectML (AMD/Intel/NVIDIA): Windows 10 1903+ (built-in)
REM
REM After installation start the server with:
REM   start.bat [options]

setlocal enabledelayedexpansion

set DEVICE=auto
set SCRIPT_DIR=%~dp0

REM --- Parse arguments -------------------------------------------------------
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    echo Usage: install.bat [--device cpu^|cuda^|directml^|auto]
    exit /b 0
)
echo Unknown argument: %~1
exit /b 1
:done_args

echo === Frigate Remote Inference Server — Windows Installer ===
echo Device: %DEVICE%
echo Install directory: %SCRIPT_DIR%
echo.

REM --- Check Python ----------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

python --version

REM --- Create virtual environment --------------------------------------------
if not exist "%SCRIPT_DIR%venv" (
    echo Creating virtual environment...
    python -m venv "%SCRIPT_DIR%venv"
)

set PIP=%SCRIPT_DIR%venv\Scripts\pip.exe
set PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe

"%PIP%" install --upgrade pip --quiet

REM --- Install core dependencies ---------------------------------------------
echo Installing core dependencies...
"%PIP%" install --quiet pyzmq numpy "opencv-python-headless>=4.8"

REM --- Install ONNX Runtime variant ------------------------------------------
if /i "%DEVICE%"=="cpu" (
    echo Installing onnxruntime ^(CPU only^)...
    "%PIP%" install --quiet "onnxruntime>=1.17"
    goto :ort_done
)

if /i "%DEVICE%"=="cuda" (
    echo Installing onnxruntime-gpu ^(CUDA^)...
    "%PIP%" install --quiet "onnxruntime-gpu>=1.17"
    goto :ort_done
)

if /i "%DEVICE%"=="directml" (
    echo Installing onnxruntime-directml ^(DirectML - AMD/Intel/NVIDIA on Windows^)...
    "%PIP%" install --quiet "onnxruntime-directml>=1.17"
    goto :ort_done
)

if /i "%DEVICE%"=="auto" (
    REM Try to detect NVIDIA GPU
    where nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        echo NVIDIA GPU detected — installing onnxruntime-gpu ^(CUDA^)...
        "%PIP%" install --quiet "onnxruntime-gpu>=1.17"
        if errorlevel 1 (
            echo onnxruntime-gpu install failed. Trying DirectML...
            "%PIP%" install --quiet "onnxruntime-directml>=1.17"
            if errorlevel 1 (
                echo DirectML install failed. Falling back to CPU...
                "%PIP%" install --quiet "onnxruntime>=1.17"
            )
        )
    ) else (
        REM No NVIDIA detected — try DirectML (works on AMD, Intel, and NVIDIA on Windows)
        echo No NVIDIA GPU detected — installing onnxruntime-directml ^(DirectML^)...
        "%PIP%" install --quiet "onnxruntime-directml>=1.17"
        if errorlevel 1 (
            echo DirectML install failed. Falling back to CPU...
            "%PIP%" install --quiet "onnxruntime>=1.17"
        )
    )
    goto :ort_done
)

echo ERROR: Unknown device '%DEVICE%'. Valid values: auto, cpu, cuda, directml
exit /b 1

:ort_done

REM --- Create start script ---------------------------------------------------
(
    echo @echo off
    echo REM start.bat — Start the Frigate remote inference server
    echo set SCRIPT_DIR=%%~dp0
    echo "%%SCRIPT_DIR%%venv\Scripts\python.exe" -m inference_server %%*
) > "%SCRIPT_DIR%start.bat"

echo.
echo === Installation complete! ===
echo.
echo Start the server:
echo   cd %SCRIPT_DIR% ^&^& start.bat
echo.
echo Common options:
echo   start.bat --endpoint tcp://*:5555
echo   start.bat --device cuda
echo   start.bat --model-dir C:\frigate-models
echo   start.bat --log-level DEBUG
echo.
echo See README.md for full configuration details.
