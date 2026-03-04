@echo off
REM Double-click to run the Strategy Lab (Windows).
REM Run SETUP first (pip install -e ".[gui]") if you haven't yet.
cd /d "%~dp0"
echo Installing or updating Strategy Lab...
python -m pip install -e ".[gui]" -q
if errorlevel 1 (
    echo.
    echo Failed. Open a terminal here and run:  pip install -e ".[gui]"
    echo See SETUP.md for full steps.
    pause
    exit /b 1
)
echo Starting Strategy Lab...
start http://localhost:8501
python -m streamlit run scripts/app.py
pause
