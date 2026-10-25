@echo off
title CurrencyGuard AI - Real-Time Counterfeit Checker
echo =====================================================================
echo Starting CurrencyGuard AI (Real-Time Counterfeit Checker)
echo =====================================================================

REM Ensure execution starts from the script's directory
cd /d "%~dp0"

REM 1. Priority: Project virtual environment (.venv)
if exist ".venv\Scripts\streamlit.exe" (
    echo [INFO] Using project virtual environment (.venv)...
    ".venv\Scripts\streamlit.exe" run app.py
    goto end
)

REM 2. Standard virtual environment (venv)
if exist "venv\Scripts\streamlit.exe" (
    echo [INFO] Using virtual environment (venv)...
    "venv\Scripts\streamlit.exe" run app.py
    goto end
)

REM 3. Check for streamlit in system PATH
where streamlit >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Launching via system streamlit command...
    streamlit run app.py
    goto end
)

REM 4. Check for system python
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Launching via python -m streamlit...
    python -m streamlit run app.py
    goto end
)

REM 5. Check for Windows Python Launcher (py)
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [INFO] Launching via py -m streamlit...
    py -m streamlit run app.py
    goto end
)

echo.
echo =====================================================================
echo [ERROR] Python or Streamlit was not found on your system!
echo =====================================================================
echo Please ensure Python 3.10 or higher is installed and added to PATH.
echo Then run:
echo   pip install -r requirements.txt
echo   streamlit run app.py
echo =====================================================================
echo.

:end
pause
