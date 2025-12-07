@echo off

py -3.10 -m venv venv

IF NOT EXIST "venv" (
    echo Virtual environment creation failed.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -m pip install --upgrade pip

IF EXIST requirements_cpu.txt (
    python -m pip install -r requirements_cpu.txt
) ELSE (
    echo requirements_cpu.txt not found!
    pause
    exit /b 1
)

python main.py

pause
