@echo off
setlocal
echo ===================================================
echo Starting QuantX Financial Intelligence Platform...
echo ===================================================

set "QUANTX_API_KEY=rc_43a33fc68f4641e598173a55da96b75af2d580ba0ad3b47a20b48c00b0582503"
set "API_KEY=rc_43a33fc68f4641e598173a55da96b75af2d580ba0ad3b47a20b48c00b0582503"

python -m pip install -r requirements.txt
if errorlevel 1 (
	echo Failed to install Python dependencies.
	pause
	exit /b 1
)

echo Starting FastAPI Uvicorn Server at http://127.0.0.1:8000 ...
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
