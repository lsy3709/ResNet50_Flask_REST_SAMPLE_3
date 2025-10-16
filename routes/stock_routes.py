from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import torch
from services.model_loader import stock_models, stock_scalers, stock_df
from config import DEVICE

stock_bp = Blueprint('stock_bp', __name__)


@stock_bp.route('/data')
def get_stockdata():
    """CSV 파일에서 주가 데이터를 가져와 JSON으로 응답합니다."""
    if stock_df is None:
        return jsonify({"error": "서버에 데이터 파일이 없습니다."}), 500

    period = request.args.get('period', '5d')
    df_copy = stock_df.copy()
    df_copy.columns = [col.capitalize() for col in df_copy.columns]

    period_map = {'1d': 1, '5d': 5}
    days = period_map.get(period, 5)
    recent_data = df_copy.tail(days)

    if period == '1d':
        data_subset = recent_data
    else:
        data_subset = recent_data.iloc[:-1]

    data_subset = data_subset.reset_index()
    data_subset['Date'] = data_subset['Date'].dt.strftime('%Y-%m-%d')
    stock_data_json = data_subset.to_dict(orient='records')

    return jsonify(stock_data_json)


@stock_bp.route('/predict/<string:model_type>', methods=['POST'])
def predict_stock(model_type):
    """
    URL 경로에서 모델 타입을 받고, JSON 본문에서 'data'를 받아 예측합니다.
    기존의 predict, predict2 라우트를 하나로 통합하고 개선했습니다.
    """
    try:
        req_data = request.get_json()
        input_data = req_data.get('data')

        if not input_data:
            return jsonify({"error": "요청 본문에 'data'가 포함되어야 합니다."}), 400

        model_key = model_type.upper()
        model = stock_models.get(model_key)
        scaler = stock_scalers.get(model_key)

        if not model or not scaler:
            return jsonify({"error": f"'{model_type}' 모델을 서버에서 찾을 수 없습니다."}), 404

        input_np = np.array(input_data)
        if input_np.ndim == 1:  # 데이터가 1차원 배열일 경우 2차원으로 변환
            input_np = input_np.reshape(1, -1)

        input_scaled = scaler.transform(input_np)
        input_tensor = torch.Tensor(input_scaled).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction_scaled = model(input_tensor).item()

        # 스케일러의 feature 개수에 맞춰 더미 배열 생성 후 inverse_transform
        num_features = scaler.n_features_in_
        dummy_array = np.zeros((1, num_features))
        dummy_array[0, -1] = prediction_scaled  # 마지막 feature가 예측값(Close)이라고 가정

        prediction = scaler.inverse_transform(dummy_array)[0][-1]

        return jsonify({"prediction": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": "예측 중 서버 오류가 발생했습니다.", "details": str(e)}), 500
