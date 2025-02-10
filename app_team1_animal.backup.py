# import os
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from torchvision import models
# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# from PIL import Image
# import io
# import json
#
# # ✅ Flask 앱 초기화
# app = Flask(__name__)
# CORS(app)  # CORS 허용
#
# # ✅ 모델 로드 (ResNet-50)
# # MODEL_PATH = "D:/workspace/수업준비/CNN_RESNET_작업/team3_tools_resnet/model_result_weight/resnet50_best_team3_tools_accuracy_90.pth"
# MODEL_PATH = "./resnet50_best_team1_animal.pth"
# NUM_CLASSES = 5  # 클래스 개수 (사용자가 설정한 클래스 수)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ✅ ResNet-50 모델 정의 및 가중치 로드
# model = models.resnet50(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
# model.eval()  # 모델을 평가 모드로 설정
#
# # ✅ 이미지 전처리 파이프라인
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # ✅ 클래스 리스트 (예제용, 사용자가 직접 정의 가능)
# class_labels = ["고양이", "공룡", "강아지", "꼬북이", "티벳여우"]
#
#
# # 🔹 1️⃣ 기본 Index 화면 (파일 업로드 UI)
# @app.route("/")
# def index():
#     return """
#     <h1>flask server Team1 Animals Classification</h1>
#     <form action="/predict" method="post" enctype="multipart/form-data">
#         <input type="file" name="image">
#         <input type="submit" value="Predict">
#     </form>
#     """
#
#
# # 🔹 2️⃣ 이미지 예측 API (POST 요청)
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "이미지가 업로드되지 않았습니다."})
#
#     file = request.files["image"]
#
#     if file.filename == "":
#         return jsonify({"error": "파일이 선택되지 않았습니다."})
#
#     try:
#         # ✅ 이미지 로드 및 변환
#         image = Image.open(io.BytesIO(file.read())).convert("RGB")
#         image = transform(image).unsqueeze(0).to(device)
#
#         # ✅ 예측 수행
#         with torch.no_grad():
#             outputs = model(image)
#             probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # 확률 변환
#             predicted_class = torch.argmax(probabilities).item()
#             confidence = probabilities[predicted_class].item() * 100  # 확률 값 (백분율)
#
#         # ✅ JSON 응답 반환
#         result = {
#             "filename": file.filename,
#             "predicted_class": class_labels[predicted_class],
#             "confidence": f"{confidence:.2f}%",
#             "class_index": predicted_class
#         }
#
#         return app.response_class(
#             response=json.dumps(result, ensure_ascii=False),  # ✅ 한글 깨짐 방지
#             status=200,
#             mimetype="application/json"
#         )
#
#     except Exception as e:
#         return app.response_class(
#             response=json.dumps({"error": str(e)}, ensure_ascii=False),
#             status=500,
#             mimetype="application/json"
#         )
# # ✅ Flask 앱 실행
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
