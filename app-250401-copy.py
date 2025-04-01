# import os
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, send_file
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from PIL import Image
# import io
# import json
# import threading
# import cv2
# import numpy as np
# import yfinance as yf
# from werkzeug.utils import secure_filename
# from ultralytics import YOLO
# import urllib.parse
# import re
# import eventlet
# import eventlet.wsgi
#
#
#
#
# # ✅ Flask 앱 초기화
# app = Flask(__name__)
# CORS(app)  # CORS 허용
#
#
# # ✅ 이미지 분류 모델 설정 (팀별)
# MODEL_CONFIGS = {
#     "team1": {
#         "model_path": "./resnet50_best_team1_animal.pth",
#         "num_classes": 5,
#         "class_labels": ["고양이", "공룡", "강아지", "꼬북이", "티벳여우"],
#     },
#     "team2": {
#         "model_path": "./resnet50_best_team2_recycle.pth",
#         "num_classes": 13,
#         "class_labels": [
#             "영업용_냉장고", "컴퓨터_cpu", "드럼_세탁기", "냉장고", "컴퓨터_그래픽카드",
#             "메인보드", "전자레인지", "컴퓨터_파워", "컴퓨터_램", "스탠드_에어컨",
#             "TV", "벽걸이_에어컨", "통돌이_세탁기"
#         ],
#     },
#     "team3": {
#         "model_path": "./resnet50_best_team3_tools_accuracy_90.pth",
#         "num_classes": 10,
#         "class_labels": [
#             "공구 톱", "공업용가위", "그라인더", "니퍼", "드라이버", "망치",
#             "스패너", "전동드릴", "줄자", "캘리퍼스"
#         ],
#     },
# }
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # ✅ YOLO 모델 로드
# yolo_model = YOLO("best.pt")
# # app.config['SERVER_NAME'] = '10.100.201.87:5000'  # Flask 서버 주소와 포트 설정
#
# # ✅ 결과 저장 폴더 설정
# UPLOAD_FOLDER = 'uploads'
# RESULT_FOLDER = 'results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
#
# processing_status = {}
#
# # 🔹 2️⃣ YOLO 비동기 처리 함수
# def process_yolo(file_path, output_path, file_type):
#     """YOLO 모델을 비동기적으로 실행"""
#     try:
#         print(f"✅ [INFO] YOLO 처리 시작 - {file_path}")
#
#         if file_type == 'image':
#             results = yolo_model(file_path)
#             result_img = results[0].plot()
#             cv2.imwrite(output_path, result_img)
#
#         elif file_type == 'video':
#             cap = cv2.VideoCapture(file_path)
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 results = yolo_model(frame)
#                 result_frame = results[0].plot()
#                 out.write(result_frame)
#
#             cap.release()
#             out.release()
#
#         print(f"✅ [INFO] YOLO 처리 완료 - {output_path}")
#
#     except Exception as e:
#         print(f"❌ [ERROR] YOLO 처리 중 오류 발생: {str(e)}")
#
# @app.route('/download/<filename>')
# def download_file(filename):
#     file_path = os.path.join(RESULT_FOLDER, filename)
#     if not os.path.isfile(file_path):
#         return jsonify({"error": "File not found"}), 404
#     return send_file(file_path, as_attachment=True, download_name=filename)
#
#
#
# # 🔹 1️⃣ 파일 업로드 API (POST 요청)
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "파일이 업로드되지 않았습니다."}), 400
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
#
#     filename = file.filename
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(file_path)
#
#     output_filename = f"result_{filename}"
#     # 결과 파일은 RESULT_FOLDER에 저장됩니다.
#     output_path = os.path.join(RESULT_FOLDER, output_filename)
#
#     print("filename : " + filename)
#     # 업로드된 파일의 확장자를 확인하여 이미지(image)인지 비디오(video)인지 판별합니다.
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#         file_type = 'image'
#     elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#         file_type = 'video'
#     else:
#         # 지원하지 않는 파일 형식일 경우 400 에러를 반환합니다.
#         return jsonify({"error": "Unsupported file type"}), 400
#
#     request_id = filename.split(".")[0]  # 파일명을 요청 ID로 사용
#
#     # YOLO 비동기 처리
#     thread = threading.Thread(target=process_yolo, args=(file_path, output_path, file_type, request_id))
#     thread.start()
#
#     # ✅ 업로드 성공 응답
#     return jsonify({
#         "message": "파일 업로드 성공",
#         "filename": filename,
#         "file_url": url_for('uploaded_file', filename=filename, _external=True)
#     }), 200
#
#
# # 🔹 2️⃣ 업로드된 파일 제공 API
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)
#
# # ✅ 이미지 분류 모델 로드 함수
# def load_model(model_type):
#     if model_type not in MODEL_CONFIGS:
#         raise ValueError(f"지원되지 않는 모델 유형: {model_type}")
#
#     config = MODEL_CONFIGS[model_type]
#
#     model = models.resnet50(pretrained=False)
#     num_ftrs = model.fc.in_features
#     model.fc = nn.Linear(num_ftrs, config["num_classes"])
#
#     model.load_state_dict(torch.load(config["model_path"], map_location=device))
#     model.to(device)
#     model.eval()
#
#     return model, config["class_labels"]
#
#
# # ✅ 이미지 전처리 파이프라인
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
# # 🔹 1️⃣ 기본 Index 화면 (파일 업로드 UI)
# @app.route("/")
# def index():
#     return render_template('index.html')
#
# # 🔹 4️⃣ 이미지 분류 API (POST 요청)
# @app.route("/predict/<model_type>", methods=["POST"])
# def predict(model_type):
#     if "image" not in request.files:
#         return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400
#
#     file = request.files["image"]
#
#     if file.filename == "":
#         return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
#
#     filename = file.filename
#     sanitized_filename = re.sub(r"[^\w.-]", "_", filename)  # 공백 및 특수문자를 _로 변경
#
#     # ✅ YOLOv8 처리 분기
#     if model_type == "yolo":
#         file_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
#         file.save(file_path)
#
#         output_filename = f"result_{sanitized_filename}"
#         output_path = os.path.join(RESULT_FOLDER, output_filename)
#
#         print(f"YOLO 처리 시작 predict_yolo , filename : {filename}")
#
#         # 파일 유형 확인
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
#             file_type = 'image'
#         elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             file_type = 'video'
#         else:
#             return jsonify({"error": "지원되지 않는 파일 형식입니다."}), 400
#
#             # ✅ YOLO 비동기 처리 (스레드 실행 후 join)
#         thread = threading.Thread(target=process_yolo, args=(file_path, output_path, file_type))
#         thread.start()
#         thread.join()  # ✅ YOLO 처리 완료될 때까지 대기
#
#         # ✅ JSON 응답으로 이미지/동영상 링크 전달
#         return jsonify({
#             "message": "YOLO 모델이 파일을 처리 중입니다.",
#             "file_url": url_for('serve_result', filename=os.path.basename(output_path), _external=True),
#             "download_url": url_for('download_file', filename=os.path.basename(output_path), _external=True),
#             "file_type": file_type,
#         })
#
#
#     # ✅ 일반 이미지 분류 처리
#     else:
#         try:
#             model, class_labels = load_model(model_type)
#
#             image = Image.open(io.BytesIO(file.read())).convert("RGB")
#             image = transform(image).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 outputs = model(image)
#                 probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
#                 predicted_class = torch.argmax(probabilities).item()
#                 confidence = probabilities[predicted_class].item() * 100
#
#             result = {
#                 "filename": file.filename,
#                 "predicted_class": class_labels[predicted_class],
#                 "confidence": f"{confidence:.2f}%",
#                 "class_index": predicted_class
#             }
#
#             return jsonify(result)
#
#         except Exception as e:
#             return jsonify({"error": str(e)}), 500
#
#
#
# # 🔹 5️⃣ 결과 파일 제공 API
# @app.route('/results/<filename>')
# def serve_result(filename):
#     """결과 파일 제공"""
#     file_path = os.path.join(RESULT_FOLDER, filename)
#
#     # ✅ 파일이 존재하는지 확인
#     if not os.path.exists(file_path):
#         return jsonify({"error": f"파일 '{filename}' 이 존재하지 않습니다."}), 404
#
#     print(f"📢 결과 파일 제공: {file_path}")  # 로그 출력
#     return send_from_directory(RESULT_FOLDER, filename)
# # ============================================================================================
# # ✅ 주식 예측 모델 정의 (RNN, LSTM, GRU)
# # 모델 및 스케일러 로드
# class StockPredictorRNN(nn.Module):  # 간단한 RNN 기반 주식 예측 모델 정의
#     def __init__(self, input_size=4, hidden_size=50, num_layers=2, output_size=1):
#         super(StockPredictorRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):  # 순전파 정의
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 은닉 상태 정의
#         out, _ = self.rnn(x, h0)  # RNN 레이어로 데이터 처리, out 모양 예시: [배치 크기, 시퀀스 길이, 은닉 크기] (예: [64, 60, 50])
#         out = self.fc(out[:, -1, :])  # 마지막 시퀀스의 은닉 상태로 최종 출력값 생성, out 모양 예시: [배치 크기, 1] (예: [64, 1])
#         return out
#
# # 모델 로드
# model = StockPredictorRNN()  # 모델 인스턴스 생성
# model.load_state_dict(torch.load('./samsungStock.pth', map_location=torch.device('cpu')))  # 모델 가중치 로드
# model.eval()  # 평가 모드로 설정
#
# # 스케일러 로드
# scaler = torch.load('./scaler.pth', map_location=torch.device('cpu'))  # 데이터 스케일러 로드
#
# #======================================================================================================================
# # 추가1: LSTM 모델 정의
# class LSTMModel(nn.Module):  # PyTorch의 LSTM 모델 클래스 정의
#     def __init__(self, input_size=4, hidden_size=128, output_size=1, num_layers=2, dropout=0.3):
#         super(LSTMModel, self).__init__()  # nn.Module의 생성자 호출
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)  # LSTM 레이어 정의
#         self.fc = nn.Linear(hidden_size, output_size)  # 완전 연결 레이어 정의
#         self.relu = nn.ReLU()  # 활성화 함수 ReLU 정의
#
#     def forward(self, x):  # 순전파 함수 정의
#         lstm_out, _ = self.lstm(x)  # LSTM의 출력 계산, lstm_out 모양 예시: [배치 크기, 시퀀스 길이, 은닉 크기] (예: [64, 60, 128])
#         last_out = lstm_out[:, -1, :]  # 마지막 시퀀스의 출력을 선택, last_out 모양 예시: [배치 크기, 은닉 크기] (예: [64, 128])
#         out = self.fc(self.relu(last_out))  # ReLU 활성화 후 완전 연결 레이어 통과, out 모양 예시: [배치 크기, 출력 크기] (예: [64, 1])
#         return out
#
# # LSTM 모델 로드
# model2 = LSTMModel()  # LSTM 모델 인스턴스 생성
# model2.load_state_dict(torch.load('./samsungStock_LSTM_60days_basic.pth', map_location=torch.device('cpu')))  # 모델 가중치 로드
# model2.eval()  # 평가 모드로 설정
# scaler = torch.load('./scaler_LSTM_60days_basic.pth')  # 스케일러 로드
#
# #======================================================================================================================
# # 추가2: GRU 모델 정의
# class GRUModel(nn.Module):  # PyTorch를 사용하여 GRU(Gated Recurrent Unit) 모델을 정의합니다
#     def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
#         super(GRUModel, self).__init__()  # nn.Module의 생성자 호출
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU 레이어 정의
#         self.fc = nn.Linear(hidden_size, output_size)  # 완전 연결 레이어 정의
#
#     def forward(self, x):  # 순전파 정의
#         out, _ = self.gru(x)  # GRU 레이어를 통해 입력 처리, out 모양 예시: [배치 크기, 시퀀스 길이, 은닉 크기] (예: [64, 60, 64])
#         out = self.fc(out[:, -1])  # 마지막 시퀀스 은닉 상태로 선형 레이어에 전달, out 모양 예시: [배치 크기, 출력 크기] (예: [64, 1])
#         return out
#
# # GRU 모델 로드
# model3 = GRUModel()  # GRU 모델 인스턴스 생성
# model3.load_state_dict(torch.load('./samsungStock_GRU.pth', map_location=torch.device('cpu')))  # 모델 가중치 로드
# model3.eval()  # 평가 모드로 설정
# scaler = torch.load('./scaler_GRU.pth')  # 스케일러 로드
#
# @app.route('/predict1', methods=['POST'])
# def predict1():  # RNN 모델 예측 엔드포인트
#     try:
#         data = request.get_json()  # JSON 데이터에서 입력 값 추출
#         if not data or 'data' not in data or 'period' not in data:  # 데이터 검증
#             return jsonify({"error": "요청에 데이터 또는 기간 정보가 포함되어 있지 않습니다."}), 400
#         input_data = data['data']
#         period = data['period']
#
#         period_days_map = {
#             '1d': 1,
#             '5d': 4,
#             '1mo': 19,
#             '3mo': 58,
#             '6mo': 116,
#             '1y': 239
#         }
#
#         if period not in period_days_map:
#             return jsonify({"error": "지원되지 않는 기간입니다."}), 400
#
#         expected_length = period_days_map[period]
#
#         if not isinstance(input_data, list) or len(input_data) != expected_length:
#             return jsonify({"error": f"잘못된 입력입니다. {expected_length}일치 Open, High, Low, Close 데이터를 제공하세요."}), 400
#
#         input_data = np.array(input_data)  # 입력 데이터 배열 생성
#         input_data = scaler.transform(input_data)  # 스케일러로 정규화
#         input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
#         input_data = torch.Tensor(input_data)
#
#         with torch.no_grad():
#             prediction = model(input_data).item()  # RNN 모델 예측 수행
#
#         prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]  # 종가 기준으로 역정규화
#
#         return jsonify({"prediction": round(prediction, 2)})
#
#     except Exception as e:
#         return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500
#
# # LSTM 예측 엔드포인트
# @app.route('/predict2', methods=['POST'])
# def predict2():
#     try:
#         data = request.get_json()  # JSON 데이터에서 입력 값 추출
#         if not data or 'data' not in data or 'period' not in data:
#             return jsonify({"error": "요청에 데이터 또는 기간 정보가 포함되어 있지 않습니다."}), 400
#         input_data = data['data']
#         period = data['period']
#
#         period_days_map = {
#             '1d': 1,
#             '5d': 4,
#             '1mo': 19,
#             '3mo': 58,
#             '6mo': 116,
#             '1y': 239
#         }
#
#         if period not in period_days_map:
#             return jsonify({"error": "지원되지 않는 기간입니다."}), 400
#
#         expected_length = period_days_map[period]
#
#         if not isinstance(input_data, list) or len(input_data) != expected_length:
#             return jsonify({"error": f"잘못된 입력입니다. {expected_length}일치 Open, High, Low, Close 데이터를 제공하세요."}), 400
#
#         input_data = np.array(input_data)  # 입력 데이터 배열 생성
#         input_data = scaler.transform(input_data)  # 스케일러로 정규화
#         input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
#         input_data = torch.Tensor(input_data)
#
#         with torch.no_grad():
#             prediction = model2(input_data).item()  # LSTM 모델 예측 수행
#
#         prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]  # 종가 기준으로 역정규화
#
#         return jsonify({"prediction": round(prediction, 2)})
#
#     except Exception as e:
#         return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500
#
# # GRU 예측 엔드포인트
# @app.route('/predict3', methods=['POST'])
# def predict3():
#     try:
#         data = request.get_json()  # JSON 데이터에서 입력 값 추출
#         if not data or 'data' not in data or 'period' not in data:
#             return jsonify({"error": "요청에 데이터 또는 기간 정보가 포함되어 있지 않습니다."}), 400
#         input_data = data['data']
#         period = data['period']
#
#         period_days_map = {
#             '1d': 1,
#             '5d': 4,
#             '1mo': 19,
#             '3mo': 58,
#             '6mo': 116,
#             '1y': 239
#         }
#
#         if period not in period_days_map:
#             return jsonify({"error": "지원되지 않는 기간입니다."}), 400
#
#         expected_length = period_days_map[period]
#
#         if not isinstance(input_data, list) or len(input_data) != expected_length:
#             return jsonify({"error": f"잘못된 입력입니다. {expected_length}일치 Open, High, Low, Close 데이터를 제공하세요."}), 400
#
#         input_data = np.array(input_data)  # 입력 데이터 배열 생성
#         input_data = scaler.transform(input_data)  # 스케일러로 정규화
#         input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가
#         input_data = torch.Tensor(input_data)
#
#         with torch.no_grad():
#             prediction = model3(input_data).item()  # GRU 모델 예측 수행
#
#         prediction = scaler.inverse_transform([[0, 0, 0, prediction]])[0][3]  # 종가 기준으로 역정규화
#
#         return jsonify({"prediction": round(prediction, 2)})
#
#     except Exception as e:
#         return jsonify({"error": "예측 중 오류가 발생했습니다.", "details": str(e)}), 500
#
# # 요청 일수에 따라 동적으로 받기
# @app.route('/get_stock_data', methods=['GET'])
# def get_stock_data():
#     period = request.args.get('period', default='5d')  # 기본 요청 기간 설정
#     ticker = '005930.KS'  # 삼성전자 종목 코드
#     data = yf.download(ticker, period=period, interval='1d')  # 지정된 기간 동안 주식 데이터 가져오기
#
#     data.columns = data.columns.get_level_values(0)  # MultiIndex가 설정된 경우 열 이름 단순화
#
#     if period == '1d':
#         data_subset = data[['Open', 'Low', 'High', 'Close']]  # 1일의 경우 데이터를 그대로 반환
#     else:
#         data_subset = data.iloc[:-1][['Open', 'Low', 'High', 'Close']]  # 나머지 기간은 최근 1일 제외하고 반환
#
#     data_subset = data_subset.reset_index()  # Date 인덱스를 컬럼으로 변환
#     data_subset['Date'] = data_subset['Date'].astype(str)  # Date를 문자열로 변환
#
#     stock_data = data_subset.to_dict(orient='records')  # JSON으로 변환 가능한 딕셔너리로 변환
#
#     return jsonify(stock_data)
#
# # ============================================================================================
#
# # ✅ Flask 실행
# if __name__ == "__main__":
#     eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 5000)), app)