@echo off
REM install_service.bat — Install the Frigate inference server as a Windows service
REM
REM Usage (run as Administrator):
REM   install_service.bat [options]
REM
REM Options:
REM   --name   <name>      Service name (default: FrigateInference)
REM   --endpoint <addr>    ZMQ endpoint (default: tcp://*:5555)
REM   --device <device>    auto|cuda|directml|cpu (default: auto)
REM   --health-port <port> Health endpoint port, 0=disabled (default: 5556)
REM   --workers <n>        Worker thread count (default: 4)
REM   --log-level <level>  DEBUG|INFO|WARNING|ERROR (default: INFO)
REM   --remove             Remove (uninstall) the service instead of installing
REM
REM Requires:
REM   - install.bat must have been run first to create the venv
REM   - Run this script as Administrator
REM
REM This script uses NSSM (Non-Sucking Service Manager) if it is on PATH.
REM If NSSM is not found, it falls back to sc.exe with a wrapper VBScript.
REM Download NSSM from https://nssm.cc/download and add it to PATH for the
REM best experience (automatic restart, stdout/stderr logging to files, etc.).

setlocal enabledelayedexpansion

REM --- Defaults --------------------------------------------------------------
set SERVICE_NAME=FrigateInference
set ENDPOINT=tcp://*:5555
set DEVICE=auto
set HEALTH_PORT=5556
set WORKERS=4
set LOG_LEVEL=INFO
set REMOVE=0
set SCRIPT_DIR=%~dp0

REM --- Parse arguments -------------------------------------------------------
:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--name"         ( set SERVICE_NAME=%~2  & shift & shift & goto :parse_args )
if /i "%~1"=="--endpoint"     ( set ENDPOINT=%~2      & shift & shift & goto :parse_args )
if /i "%~1"=="--device"       ( set DEVICE=%~2        & shift & shift & goto :parse_args )
if /i "%~1"=="--health-port"  ( set HEALTH_PORT=%~2   & shift & shift & goto :parse_args )
if /i "%~1"=="--workers"      ( set WORKERS=%~2       & shift & shift & goto :parse_args )
if /i "%~1"=="--log-level"    ( set LOG_LEVEL=%~2     & shift & shift & goto :parse_args )
if /i "%~1"=="--remove"       ( set REMOVE=1          & shift          & goto :parse_args )
if /i "%~1"=="--help" (
    echo Usage: install_service.bat [--name NAME] [--endpoint ADDR]
    echo        [--device auto^|cuda^|directml^|cpu] [--health-port PORT]
    echo        [--workers N] [--log-level LEVEL] [--remove]
    exit /b 0
)
echo Unknown argument: %~1
exit /b 1
:done_args

REM --- Check Administrator privileges ----------------------------------------
net session >nul 2>&1
if errorlevel 1 (
    echo ERROR: This script must be run as Administrator.
    echo Right-click the script and choose "Run as administrator".
    exit /b 1
)

REM --- Check that the venv exists --------------------------------------------
if not exist "%SCRIPT_DIR%venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at %SCRIPT_DIR%venv
    echo Please run install.bat first.
    exit /b 1
)

set PYTHON_EXE=%SCRIPT_DIR%venv\Scripts\python.exe
set SERVER_ARGS=--endpoint "%ENDPOINT%" --device %DEVICE% --health-port %HEALTH_PORT% --workers %WORKERS% --log-level %LOG_LEVEL%

REM ============================================================================
REM  REMOVE MODE
REM ============================================================================
if %REMOVE%==1 (
    echo Removing service %SERVICE_NAME%...

    where nssm >nul 2>&1
    if not errorlevel 1 (
        nssm stop  %SERVICE_NAME% >nul 2>&1
        nssm remove %SERVICE_NAME% confirm
    ) else (
        net stop "%SERVICE_NAME%" >nul 2>&1
        sc delete "%SERVICE_NAME%"
    )
    echo Service %SERVICE_NAME% removed.
    exit /b 0
)

REM ============================================================================
REM  INSTALL MODE
REM ============================================================================
echo === Frigate Inference Server — Windows Service Installer ===
echo Service name  : %SERVICE_NAME%
echo Python        : %PYTHON_EXE%
echo Arguments     : -m inference_server %SERVER_ARGS%
echo.

REM --- Prefer NSSM -----------------------------------------------------------
where nssm >nul 2>&1
if not errorlevel 1 (
    echo Using NSSM to create service...

    REM Stop and remove any existing service with the same name
    nssm stop   %SERVICE_NAME% >nul 2>&1
    nssm remove %SERVICE_NAME% confirm >nul 2>&1

    nssm install %SERVICE_NAME% "%PYTHON_EXE%" "-m inference_server %SERVER_ARGS%"
    nssm set %SERVICE_NAME% AppDirectory "%SCRIPT_DIR%"

    REM Route stdout and stderr to log files next to the script
    set LOG_DIR=%SCRIPT_DIR%logs
    if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
    nssm set %SERVICE_NAME% AppStdout "%LOG_DIR%\inference_server.log"
    nssm set %SERVICE_NAME% AppStderr "%LOG_DIR%\inference_server_err.log"
    nssm set %SERVICE_NAME% AppRotateFiles 1
    nssm set %SERVICE_NAME% AppRotateBytes 10485760

    REM Restart automatically on failure
    nssm set %SERVICE_NAME% AppExit Default Restart
    nssm set %SERVICE_NAME% AppRestartDelay 5000

    REM Start automatically at boot
    sc config "%SERVICE_NAME%" start= auto

    net start "%SERVICE_NAME%"
    if errorlevel 1 (
        echo WARNING: Service installed but failed to start immediately.
        echo Check logs in %LOG_DIR%
    ) else (
        echo Service %SERVICE_NAME% started successfully.
    )
    goto :install_done
)

REM --- Fallback: sc.exe + wrapper script -------------------------------------
echo NSSM not found — using sc.exe with a VBScript wrapper.
echo (Download NSSM from https://nssm.cc for better logging and restart support.)
echo.

REM Create a VBScript wrapper that runs python.exe without a console window.
set WRAPPER=%SCRIPT_DIR%run_service.vbs
(
    echo Set WshShell = CreateObject^("WScript.Shell"^)
    echo WshShell.Run """%PYTHON_EXE%"" -m inference_server %SERVER_ARGS%", 0, False
) > "%WRAPPER%"

REM sc.exe needs a quoted binPath with the full path to wscript.exe and the script
set WSCRIPT=%SystemRoot%\System32\wscript.exe
sc create "%SERVICE_NAME%" binPath= "\"%WSCRIPT%\" \"%WRAPPER%\"" start= auto DisplayName= "Frigate Inference Server"
if errorlevel 1 (
    echo ERROR: sc create failed. See above for details.
    exit /b 1
)

REM Set a description
sc description "%SERVICE_NAME%" "Frigate remote GPU inference server (ZMQ + ONNX Runtime)"

net start "%SERVICE_NAME%"
if errorlevel 1 (
    echo WARNING: Service installed but failed to start immediately.
    echo Try: net start %SERVICE_NAME%
) else (
    echo Service %SERVICE_NAME% started successfully.
)

:install_done
echo.
echo === Done! ===
echo.
echo Useful commands:
echo   sc query %SERVICE_NAME%           (check status)
echo   net start %SERVICE_NAME%          (start)
echo   net stop  %SERVICE_NAME%          (stop)
echo   install_service.bat --remove      (uninstall)
echo.
echo To view logs (NSSM only):
echo   type "%SCRIPT_DIR%logs\inference_server.log"
