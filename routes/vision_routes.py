import os
import uuid
import threading
from flask import Blueprint, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import io
import torchvision.transforms as transforms
from services.model_loader import yolo_model, load_vision_model
from services.yolo_processor import process_yolo
from config import UPLOAD_FOLDER, RESULT_FOLDER, DEVICE

vision_bp = Blueprint('vision_bp', __name__)


@vision_bp.route("/predict/<model_type>", methods=["POST"])
def predict_vision(model_type):
    if "image" not in request.files:
        return jsonify({"error": "이미지가 업로드되지 않았습니다."}), 400

    file = request.files["image"]
    original_filename = file.filename
    if original_filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    # --- YOLO 모델 처리 ---
    if model_type == 'yolo':
        return handle_yolo_prediction(file, original_filename)

    # --- 이미지 분류 모델 처리 ---
    else:
        return handle_classification(file, model_type, original_filename)


def handle_yolo_prediction(file, original_filename):
    """YOLO 예측 요청을 처리하고 비동기 작업을 시작합니다."""
    filename_base, file_extension = os.path.splitext(original_filename)

    if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        file_type = 'image'
    elif file_extension.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        file_type = 'video'
    else:
        return jsonify({"error": "지원되지 않는 파일 형식입니다."}), 400

    safe_filename_base = secure_filename(filename_base)
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{safe_filename_base}_{unique_id}{file_extension}"

    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(file_path)

    output_filename = f"result_{safe_filename_base}_{unique_id}{'.mp4' if file_type == 'video' else file_extension}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    thread = threading.Thread(target=process_yolo, args=(file_path, output_path, file_type))
    thread.start()

    # url_for의 _external=True 옵션으로 전체 URL을 생성합니다.
    status_url = url_for('vision_bp.get_status', filename=output_filename, _external=True)

    return jsonify({
        "message": "YOLO 처리가 시작되었습니다.",
        "output_filename": output_filename,
        "status_url": status_url
    })


def handle_classification(file, model_type, original_filename):
    """이미지 분류 예측을 처리합니다."""
    try:
        model, class_labels = load_vision_model(model_type)
        if model is None:
            return jsonify({"error": f"지원되지 않는 모델 유형: {model_type}"}), 400

        image = Image.open(io.BytesIO(file.read())).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item() * 100

        return jsonify({
            "filename": original_filename,
            "predicted_class": class_labels[predicted_class_idx],
            "confidence": f"{confidence:.2f}%",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@vision_bp.route('/status/<filename>')
def get_status(filename):
    """YOLO 처리 상태를 확인하는 API (폴링용)"""
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return jsonify({
            "status": "complete",
            "url": url_for('main_bp.serve_result', filename=filename, _external=True)
        })
    else:
        return jsonify({"status": "processing"})
