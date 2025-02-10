import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import json

# ✅ Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 허용

# ✅ 모델 설정 (모델별 가중치 파일, 클래스 개수, 클래스명)
MODEL_CONFIGS = {
    "team1": {
        "model_path": "./resnet50_best_team1_animal.pth",
        "num_classes": 5,
        "class_labels": ["고양이", "공룡", "강아지", "꼬북이", "티벳여우"],
    },
    "team2": {
        "model_path": "./resnet50_best_team2_recycle.pth",
        "num_classes": 13,
        "class_labels": [
            "영업용_냉장고", "컴퓨터_cpu", "드럼_세탁기", "냉장고", "컴퓨터_그래픽카드",
            "메인보드", "전자레인지", "컴퓨터_파워", "컴퓨터_램", "스탠드_에어컨",
            "TV", "벽걸이_에어컨", "통돌이_세탁기"
        ],
    },
    "team3": {
        "model_path": "./resnet50_best_team3_tools_accuracy_90.pth",
        "num_classes": 10,
        "class_labels": [
            "공구 톱", "공업용가위", "그라인더", "니퍼", "드라이버", "망치",
            "스패너", "전동드릴", "줄자", "캘리퍼스"
        ],
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ✅ 모델 로드 함수
def load_model(model_type):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"지원되지 않는 모델 유형: {model_type}")

    config = MODEL_CONFIGS[model_type]

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config["num_classes"])

    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.to(device)
    model.eval()

    return model, config["class_labels"]


# ✅ 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 🔹 1️⃣ 기본 Index 화면 (파일 업로드 UI)
@app.route("/")
def index():
    return """
    <h1>Flask Server - Unified Image Classification</h1>
    <p>팀별 모델을 사용하여 이미지 분류를 수행할 수 있습니다.</p>
    <p>팀별 모델 유형: team1 (동물), team2 (재활용), team3 (공구)</p>
    <p>예측 요청 예시: POST /predict/team1</p>
    <form action="/predict/team1" method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Predict (Team1)">
    </form>
    """


# 🔹 2️⃣ 이미지 예측 API (POST 요청)
@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):
    if model_type not in MODEL_CONFIGS:
        return jsonify({"error": f"지원되지 않는 모델 유형: {model_type}"}), 400

    if "image" not in request.files:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    try:
        # ✅ 모델 로드
        model, class_labels = load_model(model_type)

        # ✅ 이미지 로드 및 변환
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # ✅ 예측 수행
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # 확률 변환
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item() * 100  # 확률 값 (백분율)

        # ✅ JSON 응답 반환
        result = {
            "filename": file.filename,
            "predicted_class": class_labels[predicted_class],
            "confidence": f"{confidence:.2f}%",
            "class_index": predicted_class
        }

        return app.response_class(
            response=json.dumps(result, ensure_ascii=False),  # ✅ 한글 깨짐 방지
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        return app.response_class(
            response=json.dumps({"error": str(e)}, ensure_ascii=False),
            status=500,
            mimetype="application/json"
        )


# ✅ Flask 앱 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
