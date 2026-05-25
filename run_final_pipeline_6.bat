@echo off
chcp 65001 > nul

echo ==========================================
echo  Whisper LoRA Improved Pipeline Start
echo ==========================================

cd /d C:\Github\cstool

echo.
echo [1/9] Activate conda environment
CALL C:\Users\eorhk\anaconda3\Scripts\activate.bat cs_asr2

if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment.
    pause
    exit /b 1
)

echo.
echo [8/9] Run batch pipeline test
python src\test_pipeline_batch.py

if errorlevel 1 (
    echo [ERROR] test_pipeline_batch.py failed.
    pause
    exit /b 1
)

echo.
echo [9/9] Analyze pipeline results
python src\analyze_pipeline_results.py

if errorlevel 1 (
    echo [ERROR] analyze_pipeline_results.py failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  Improved Pipeline Complete!
echo ==========================================
echo.
echo Check outputs:
echo C:\Github\cstool\outputs\whisper_lora_improved\final
echo.
echo Check results:
echo C:\Github\cstool\results
echo.

pause