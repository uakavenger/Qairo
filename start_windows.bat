@echo off
cd /d "%~dp0"
py -3 launch.py
if errorlevel 1 (
  echo.
  echo Python 3.10 or newer is required. Install Python from python.org, then try again.
  pause
)
