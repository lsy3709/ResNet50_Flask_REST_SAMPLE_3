import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from PIL import Image
import io
import json
import threading
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from flask import  request, send_file

# ✅ Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 허용
socketio = SocketIO(app, cors_allowed_origins="*")


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ✅ YOLO 모델 로드
yolo_model = YOLO("best.pt")
app.config['SERVER_NAME'] = '127.0.0.1:5000'  # Flask 서버 주소와 포트 설정

# ✅ 결과 저장 폴더 설정
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 🔹 2️⃣ YOLO 비동기 처리 함수
def process_yolo(file_path, output_path, file_type):
    with app.app_context():
        if file_type == 'image':
            results = yolo_model(file_path)
            result_img = results[0].plot()
            cv2.imwrite(output_path, result_img)

            # ✅ 이미지 결과를 실시간으로 전송 (화면에서 즉시 표시 가능)
            socketio.emit(
                'file_processed',
                {
                    'url': url_for('serve_result', filename=os.path.basename(output_path), _external=True),
                    'download_url': url_for('download_file', filename=os.path.basename(output_path), _external=True),
                 'type': 'image'}
            )
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

            # ✅ Flask 컨텍스트 내에서 URL 생성
            # 처리 완료 알림
            socketio.emit(
                'file_processed',
                {'url': url_for('download_file', filename=os.path.basename(output_path), _external=True)}
            )

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, as_attachment=True, download_name=filename)



# 🔹 1️⃣ 파일 업로드 API (POST 요청)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    output_filename = f"result_{filename}"
    # 결과 파일은 RESULT_FOLDER에 저장됩니다.
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    print("filename : " + filename)
    # 업로드된 파일의 확장자를 확인하여 이미지(image)인지 비디오(video)인지 판별합니다.
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        file_type = 'image'
    elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        file_type = 'video'
    else:
        # 지원하지 않는 파일 형식일 경우 400 에러를 반환합니다.
        return jsonify({"error": "Unsupported file type"}), 400

    # thread = threading.Thread(target=process_yolo, args=(file_path, output_path, file_type))
    # # 새로운 스레드에서 처리 작업을 시작합니다.
    # thread.start()

    # ✅ 업로드 성공 응답
    # return jsonify({
    #     "message": "파일 업로드 성공",
    #     "filename": filename,
    #     "file_url": url_for('uploaded_file', filename=filename, _external=True)
    # }), 200
    return jsonify({"message": "Processing started"})

# 🔹 2️⃣ 업로드된 파일 제공 API
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ✅ 이미지 분류 모델 로드 함수
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
    return render_template('index.html')




# 🔹 3️⃣ YOLO 예측 API (POST 요청)
@app.route("/predict2/yolo8", methods=["POST"])
def predict_yolo():
    if "image" not in request.files:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    file = request.files["image"]
    print("predict_yolo , file : " , file.filename)
    if file.filename == "":
        print("🔴 ERROR: 파일명이 비어 있습니다.")  # 디버깅 로그 추가
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    output_filename = f"result_{filename}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    print("predict_yolo , filename : " + filename)

    # 파일 유형 확인
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        file_type = 'image'
    elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        file_type = 'video'
    else:
        return jsonify({"error": "지원되지 않는 파일 형식입니다."}), 400

    # YOLO 비동기 처리
    thread = threading.Thread(target=process_yolo, args=(file_path, output_path, file_type))
    thread.start()

    return jsonify({"message": "YOLO 모델이 파일을 처리 중입니다."})


# 🔹 4️⃣ 이미지 분류 API (POST 요청)
@app.route("/predict/<model_type>", methods=["POST"])
def predict(model_type):
    if model_type not in MODEL_CONFIGS:
        return jsonify({"error": f"지원되지 않는 모델 유형: {model_type}"}), 400

    if "image" not in request.files:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    file = request.files["image"]

    # print("predict(model_type):, /predict/<model_type> , file : " + file)
    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    try:
        model, class_labels = load_model(model_type)

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


# 🔹 5️⃣ 결과 파일 제공 API
@app.route('/results/<filename>')
def serve_result(filename):
    """결과 파일 제공"""
    file_path = os.path.join(RESULT_FOLDER, filename)

    # ✅ 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        return jsonify({"error": "파일이 존재하지 않습니다."}), 404

    print(f"📢 결과 파일 제공: {file_path}")  # 로그 출력
    return send_from_directory(RESULT_FOLDER, filename)


# ✅ Flask 실행
if __name__ == "__main__":
    socketio.run(app, debug=True)
