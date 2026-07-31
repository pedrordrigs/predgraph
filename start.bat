@echo off
REM PredGraph launcher: starts the collector and the dashboard, opens the browser.
REM Close either window to stop that piece; both are safe to restart at any time.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [PredGraph] No virtualenv found. Run setup.bat first.
  pause
  exit /b 1
)

echo [PredGraph] Starting collector ^(polls markets, rediscovers hourly^)...
start "PredGraph collector" .venv\Scripts\python.exe -m predgraph.cli run

echo [PredGraph] Starting dashboard on http://127.0.0.1:8765 ...
start "PredGraph dashboard" .venv\Scripts\python.exe -m predgraph.cli web

echo.
echo [PredGraph] Both windows are running.
echo   Collector : keeps price history flowing. Leave it open.
echo   Dashboard : http://127.0.0.1:8765
echo.
echo This window can be closed.
timeout /t 8 >nul
