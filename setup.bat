@echo off
REM One-time setup: virtualenv, dependencies, database, ontology, first discovery.
REM Safe to re-run — every step is idempotent.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [Snapback] Creating virtualenv...
  py -3.12 -m venv .venv || (echo Could not create venv. Is Python 3.12 installed? & pause & exit /b 1)
)

echo [Snapback] Installing dependencies...
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.venv\Scripts\python.exe -m pip install --quiet -e .
.venv\Scripts\python.exe -m pip install --quiet fastapi uvicorn

if not exist ".env" copy ".env.example" ".env" >nul

echo [Snapback] Creating database and loading ontology...
.venv\Scripts\python.exe -m snapback.cli db init
.venv\Scripts\python.exe -m snapback.cli ontology sync

echo [Snapback] Discovering markets (takes a minute)...
.venv\Scripts\python.exe -m snapback.cli markets discover

echo.
echo [Snapback] Setup complete. Run start.bat to launch the collector and dashboard.
pause
