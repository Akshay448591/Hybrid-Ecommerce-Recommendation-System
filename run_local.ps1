# HYBRID V2 - Local Execution Script (PowerShell)
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "   Starting HYBRID V2 E-Commerce Platform & Major Project Server        " -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking / Installing dependencies..." -ForegroundColor Yellow
pip install -r backend/requirements.txt

Write-Host ""
Write-Host "Starting FastAPI Application with React frontend on http://localhost:8000 ..." -ForegroundColor Green
Write-Host "Open your browser at: http://localhost:8000" -ForegroundColor White
Write-Host "API Documentation available at: http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""

python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
