# import os
# import io
# import re
# import uuid
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from torchvision import models, transforms
# from torch import nn
# from ultralytics import YOLO
# from concurrent.futures import ThreadPoolExecutor
#
# app = FastAPI()
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
# RESULT_FOLDER = os.path.join(BASE_DIR, 'results')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# classification_models = {}
# yolo_model = YOLO("best-busanit501-aqua.pt")
# yolo_executor = ThreadPoolExecutor(max_workers=2)
#
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
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# def load_model(model_type):
#     config = MODEL_CONFIGS[model_type]
#     model = models.resnet50(pretrained=False)
#     model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
#     model.load_state_dict(torch.load(config["model_path"], map_location=device))
#     model.to(device)
#     model.eval()
#     return model, config["class_labels"]
#
# def get_classification_model(model_type):
#     if model_type not in classification_models:
#         model, class_labels = load_model(model_type)
#         classification_models[model_type] = (model, class_labels)
#     return classification_models[model_type]
#
# def process_yolo(file_path, output_path, file_type):
#     try:
#         if file_type == 'image':
#             results = yolo_model(file_path)
#             result_img = results[0].plot()
#             cv2.imwrite(output_path, result_img)
#         elif file_type == 'video':
#             cap = cv2.VideoCapture(file_path)
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 results = yolo_model(frame)
#                 result_frame = results[0].plot()
#                 out.write(result_frame)
#             cap.release()
#             out.release()
#     except Exception as e:
#         print(f"YOLO 처리 중 오류 발생: {str(e)}")
#
# @app.post("/predict/{model_type}")
# async def predict(model_type: str, image: UploadFile = File(...)):
#     filename = image.filename
#     if not filename:
#         raise HTTPException(status_code=400, detail="파일이 선택되지 않았습니다.")
#
#     sanitized_filename = re.sub(r"[^\w.-]", "_", filename)
#     unique_id = str(uuid.uuid4())
#     sanitized_filename = f"{unique_id}_{sanitized_filename}"
#     file_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
#     output_filename = f"result_{sanitized_filename}"
#     output_path = os.path.join(RESULT_FOLDER, output_filename)
#
#     contents = await image.read()
#     with open(file_path, "wb") as f:
#         f.write(contents)
#
#     if model_type == "yolo":
#         file_type = 'image' if sanitized_filename.lower().endswith(('.jpg', '.jpeg', '.png')) else 'video'
#         yolo_executor.submit(process_yolo, file_path, output_path, file_type)
#         return {
#             "message": "YOLO 모델이 파일을 처리 중입니다.",
#             "file_url": f"/results/{output_filename}",
#             "download_url": f"/download/{output_filename}",
#             "file_type": file_type,
#             "status": "processing"
#         }
#     else:
#         model, class_labels = get_classification_model(model_type)
#         image_data = Image.open(io.BytesIO(contents)).convert("RGB")
#         image_tensor = transform(image_data).unsqueeze(0).to(device)
#         with torch.no_grad():
#             outputs = model(image_tensor)
#             probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
#             predicted_class = torch.argmax(probabilities).item()
#             confidence = probabilities[predicted_class].item() * 100
#
#         return {
#             "filename": filename,
#             "predicted_class": class_labels[predicted_class],
#             "confidence": f"{confidence:.2f}%",
#             "class_index": predicted_class
#         }
#
# @app.get("/results/{filename}")
# def serve_result(filename: str):
#     file_path = os.path.join(RESULT_FOLDER, filename)
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="파일이 존재하지 않습니다.")
#     return FileResponse(file_path)
#
# @app.get("/download/{filename}")
# def download_file(filename: str):
#     file_path = os.path.join(RESULT_FOLDER, filename)
#     if not os.path.isfile(file_path):
#         raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
#     return FileResponse(file_path, filename=filename, media_type='application/octet-stream')
