## FastAPI YOLO/ResNet 서버

간단히 실행할 수 있는 FastAPI 기반 추론 서버입니다. YOLO(v8) 감지 및 ResNet50 분류 모델을 제공합니다.

### 요구 사항
- Python 3.10
- CUDA(Optional)

### 설치
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --disable-pip-version-check
```

### 실행
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 헬스 체크
```bash
curl http://localhost:8000/health
```

### 예측 요청 (YOLO)
```bash
curl -X POST "http://localhost:8000/predict/yolo" \
  -F "image=@uploads/3ea70b5c-f9b1-4893-b48a-e944e0eb56d4_test1.jpg"
```
- 응답 예: `{"url":"http://localhost:8000/results/result_<uuid>_<filename>"}`

### 결과 이미지 받기
```bash
curl -L -o result.jpg "http://localhost:8000/results/result_<uuid>_<filename>"
```

### 프론트엔드(React) 연동 팁
- 서버가 반환하는 전체 URL(`url` 또는 `file_url`)을 그대로 사용하세요.
- 파일 생성 지연을 대비해 1~2초 폴링 또는 쿼리파라미터로 캐시 무효화(`?t=timestamp`).
- 본 저장소 서버는 `/results/{filename}`에서 최대 15초 대기 후 파일을 응답합니다.

### 트러블슈팅
- 404 Not Found (결과 파일): 업로드 직후 접근 시 생성이 끝나지 않았을 수 있습니다. 잠시 뒤 재시도하세요.
- Windows에서 파워셸 cURL과 Git Bash cURL 동작이 다를 수 있습니다. 필요 시 `curl.exe`를 명시적으로 사용하세요.

### 라이선스
내부 프로젝트 샘플 코드
