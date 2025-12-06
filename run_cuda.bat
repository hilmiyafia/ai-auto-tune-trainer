@echo off

chcp 65001 >nul

echo(
echo ╔══════════════════╗
echo ║ 0/3 INITIALIZING ║
echo ╚══════════════════╝
echo(

py -3.10 -m venv venv

IF NOT EXIST "venv" (
    echo Virtual environment creation failed.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -m pip install --upgrade pip

python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

IF EXIST requirements_cuda.txt (
    python -m pip install -r requirements_cuda.txt
) ELSE (
    echo requirements_cuda.txt not found!
    pause
    exit /b 1
)

python run.py

pause
