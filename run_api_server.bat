@echo off
chcp 65001 > nul

cd /d C:\Github\cstool

set PYTHON_EXE=C:\Users\eorhk\anaconda3\envs\cs_asr2\python.exe

echo ==========================================
echo  CSTOOL FastAPI Backend Start
echo ==========================================

%PYTHON_EXE% -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000

pause