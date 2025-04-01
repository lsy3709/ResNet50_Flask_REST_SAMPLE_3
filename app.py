# app.py (리팩토링 반영)
import os
import io
import json
import threading
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, request, jsonify, send_file, send_from_directory, url_for, render_template
from flask_cors import CORS
from ultralytics import YOLO
import yfinance as yf
from werkzeug.utils import secure_filename
import eventlet
import eventlet.wsgi

import uuid
# ========== Flask 초기화 및 설정 ==========
app = Flask(__name__)
CORS(app)
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ========== 디바이스 설정 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 모델 전역 변수 선언 ==========
rnn_model = None
lstm_model = None
gru_model = None
scalers = {}
yolo_model = YOLO("best.pt")
classification_models = {}
yolo_executor = ThreadPoolExecutor(max_workers=2)  # YOLO 작업용 제한

# ========== 모델 정의 ==========
class StockPredictorRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 50).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=1, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(self.relu(last_out))

class GRUModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

# ========== 모델 및 스케일러 초기화 함수 ==========
def initialize_models():
    global rnn_model, lstm_model, gru_model, scalers

    rnn_model = StockPredictorRNN()
    rnn_model.load_state_dict(torch.load('./samsungStock.pth', map_location=device))
    rnn_model.eval()

    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load('./samsungStock_LSTM_60days_basic.pth', map_location=device))
    lstm_model.eval()

    gru_model = GRUModel()
    gru_model.load_state_dict(torch.load('./samsungStock_GRU.pth', map_location=device))
    gru_model.eval()

    scalers['rnn'] = torch.load('./scaler.pth')
    scalers['lstm'] = torch.load('./scaler_LSTM_60days_basic.pth')
    scalers['gru'] = torch.load('./scaler_GRU.pth')

# ✅ 이미지 분류 모델 설정 (팀별)
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

# ✅ 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🔹 YOLO 처리 함수

def process_yolo(file_path, output_path, file_type):
    try:
        if file_type == 'image':
            results = yolo_model(file_path)
            result_img = results[0].plot()
            cv2.imwrite(output_path, result_img)

        elif file_type == 'video':
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model(frame)
                result_frame = results[0].plot()
                out.write(result_frame)

            cap.release()
            out.release()

    except Exception as e:
        print(f"YOLO 처리 중 오류 발생: {str(e)}")

# 🔹 이미지 분류 모델 로드 함수

def load_model(model_type):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"지원되지 않는 모델 유형: {model_type}")

    config = MODEL_CONFIGS[model_type]
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.to(device)
    model.eval()

    return model, config["class_labels"]

# 🔹 이미지 분류 모델 캐싱 함수

def get_classification_model(model_type):
    if model_type not in classification_models:
        model, class_labels = load_model(model_type)
        classification_models[model_type] = (model, class_labels)
    return classification_models[model_type]

 # 🔹 4️⃣ 이미지 분류 API (POST 요청)
@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):
    if "image" not in request.files:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    filename = file.filename
    sanitized_filename = re.sub(r"[^\w.-]", "_", filename)  # 공백 및 특수문자를 _로 변경
    unique_id = str(uuid.uuid4())
    sanitized_filename = f"{unique_id}_{sanitized_filename}"

    # ✅ YOLOv8 처리 분기
    if model_type == "yolo":
        file_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
        file.save(file_path)

        output_filename = f"result_{sanitized_filename}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        print(f"YOLO 처리 시작 predict_yolo , filename : {sanitized_filename}")

        # 파일 유형 확인
        if sanitized_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_type = 'image'
        elif sanitized_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            file_type = 'video'
        else:
            return jsonify({"error": "지원되지 않는 파일 형식입니다."}), 400

            # ✅ YOLO 비동기 처리 (스레드 실행 후 join)
        # thread = threading.Thread(target=process_yolo, args=(file_path, output_path, file_type))
        # thread.start()
        # thread.join()  # ✅ YOLO 처리 완료될 때까지 대기
        yolo_executor.submit(process_yolo, file_path, output_path, file_type)

        # ✅ JSON 응답으로 이미지/동영상 링크 전달
        return jsonify({
            "message": "YOLO 모델이 파일을 처리 중입니다.",
            "file_url": url_for('serve_result', filename=os.path.basename(output_path), _external=True),
            "download_url": url_for('download_file', filename=os.path.basename(output_path), _external=True),
            "file_type": file_type,
            "status": "processing"
        })


    # ✅ 일반 이미지 분류 처리
    else:
        try:
            # model, class_labels = load_model(model_type)
            model, class_labels = get_classification_model(model_type)

            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item() * 100

            result = {
                "filename": file.filename,
                "predicted_class": class_labels[predicted_class],
                "confidence": f"{confidence:.2f}%",
                "class_index": predicted_class
            }

            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# 🔹 결과 제공 API
@app.route('/results/<filename>')
def serve_result(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"파일 '{filename}' 이 존재하지 않습니다."}), 404
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)

# ========== Flask 실행 ==========
if __name__ == "__main__":
    initialize_models()
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)
