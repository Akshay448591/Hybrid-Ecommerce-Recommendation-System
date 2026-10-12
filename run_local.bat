@echo off
title HYBRID V2 - E-Commerce Recommendation Engine
echo =========================================================================
echo    Starting HYBRID V2 E-Commerce Platform & Major Project Server
echo =========================================================================
echo.
echo Installing dependencies if needed...
pip install -r backend\requirements.txt
echo.
echo Starting FastAPI application with React frontend on http://localhost:8000 ...
echo Press Ctrl+C to terminate the server.
echo.
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
