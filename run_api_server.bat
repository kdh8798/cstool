@echo off
chcp 65001 > nul

echo ==========================================
echo  CSTOOL FastAPI Server Start
echo ==========================================

cd /d C:\Github\cstool

set PYTHON_EXE=C:\Users\eorhk\anaconda3\envs\cs_asr2\python.exe
set HOST=0.0.0.0
set PORT=8000

echo.
echo [1/4] Check Python environment
%PYTHON_EXE% -c "import sys; print(sys.executable)"

if errorlevel 1 (
    echo [ERROR] Python environment check failed.
    pause
    exit /b 1
)

echo.
echo [2/4] Check model files

if not exist outputs\whisper_lora_improved\final\adapter_config.json (
    echo [ERROR] adapter_config.json not found.
    echo Check: outputs\whisper_lora_improved\final
    pause
    exit /b 1
)

if not exist outputs\whisper_lora_improved\final\adapter_model.safetensors (
    echo [ERROR] adapter_model.safetensors not found.
    echo Check: outputs\whisper_lora_improved\final
    pause
    exit /b 1
)

echo Improved LoRA model found.

echo.
echo [3/4] Check FastAPI dependencies
%PYTHON_EXE% -c "import fastapi, uvicorn, multipart; print('FastAPI dependencies OK')"

if errorlevel 1 (
    echo [ERROR] FastAPI dependencies missing.
    echo Run:
    echo %PYTHON_EXE% -m pip install fastapi uvicorn python-multipart
    pause
    exit /b 1
)

echo.
echo [4/4] Start API server
echo.
echo Local docs:
echo http://127.0.0.1:%PORT%/docs
echo.
echo External access:
echo http://YOUR_PUBLIC_IP:%PORT%/docs
echo.
echo API endpoint:
echo http://YOUR_PUBLIC_IP:%PORT%/api/transcribe
echo.
echo Press Ctrl + C to stop server.
echo ==========================================

%PYTHON_EXE% -m uvicorn src.web_api:app --host %HOST% --port %PORT%

pause