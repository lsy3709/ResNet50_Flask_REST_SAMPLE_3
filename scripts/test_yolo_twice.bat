@echo off
REM 두 번 연속 업로드 후 결과 접근 테스트 (Windows)
setlocal enabledelayedexpansion
cd /d %~dp0\..

set FILE1=uploads\3ea70b5c-f9b1-4893-b48a-e944e0eb56d4_test1.jpg
set FILE2=uploads\67ba4f53-23bd-4fbf-bdde-049dca0ac700_test2.jpg

if not exist "%FILE1%" (
  echo Sample not found: %FILE1%
  goto :eof
)
if not exist "%FILE2%" (
  echo Sample not found: %FILE2%
  goto :eof
)

for %%F in ("%FILE1%" "%FILE2%") do (
  for /f "usebackq tokens=*" %%R in (`curl.exe -s -X POST "http://localhost:8000/predict/yolo" -F "image=@%%~fF"`) do set RESP=%%R
  echo RESP=!RESP!
  for /f "tokens=2 delims=:,\" }" %%U in ("!RESP!") do set URL=%%U
  echo URL=!URL!
  curl.exe -s -o NUL -w "HTTP=%{http_code}\n" !URL!
)

endlocal


