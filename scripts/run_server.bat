@echo off
REM FastAPI 서버 실행 스크립트 (Windows)
cd /d %~dp0\..
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --disable-pip-version-check
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload


