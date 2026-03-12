@echo off
REM One-time setup: install Phi-nance and the Strategy Lab (Windows).
REM Double-click this file, or run it from a terminal in this folder.
cd /d "%~dp0"
echo.
echo Phi-nance setup — installing project and GUI...
echo.
python -m pip install -e ".[gui]"
if errorlevel 1 (
    echo.
    echo Install failed. Make sure Python 3.10+ is installed and "Add to PATH" was checked.
    echo See SETUP.md for help.
    pause
    exit /b 1
)
echo.
echo Setup complete. To start the Strategy Lab, double-click: run_strategy_lab.bat
echo Or in a terminal run: streamlit run scripts/app.py
echo.
pause
