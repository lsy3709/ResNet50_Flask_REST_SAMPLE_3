import torch
import torch.nn as nn
from torchvision import models as vision_models_lib  # alias to avoid confusion
from ultralytics import YOLO
import pandas as pd
from config import VISION_MODEL_CONFIGS, YOLO_MODEL_PATH, STOCK_MODEL_PATHS, STOCK_CSV_PATH, DEVICE


# --- 주가 예측 모델 클래스 정의 ---
class StockPredictorRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
        super(StockPredictorRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=1, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(self.relu(lstm_out[:, -1, :]))


class GRUModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


# --- 모델 로딩 ---
# 앱 시작 시 한 번만 모델을 로드하여 메모리에 보관합니다.

# 1. YOLO 모델 로드
print("... YOLO 모델 로딩 중 ...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO 모델 로딩 완료")


# 2. 이미지 분류 모델 로드
def load_vision_model(model_type):
    """요청 시 특정 이미지 분류 모델을 로드합니다."""
    config = VISION_MODEL_CONFIGS.get(model_type)
    if not config:
        return None, None

    try:
        model = vision_models_lib.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config["num_classes"])
        model.load_state_dict(torch.load(config["model_path"], map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model, config["class_labels"]
    except FileNotFoundError:
        print(f"🔴 ERROR: {model_type} 모델 파일을 찾을 수 없습니다: {config['model_path']}")
        return None, None


# 3. 주가 예측 모델 로드
stock_models = {}
stock_scalers = {}

print("... 주가 예측 모델 로딩 중 ...")
for model_name, paths in STOCK_MODEL_PATHS.items():
    try:
        if model_name == 'RNN':
            stock_models[model_name] = StockPredictorRNN()
        elif model_name == 'LSTM':
            stock_models[model_name] = LSTMModel()
        elif model_name == 'GRU':
            stock_models[model_name] = GRUModel()

        stock_models[model_name].load_state_dict(torch.load(paths['model'], map_location=DEVICE))
        stock_models[model_name].eval()
        stock_scalers[model_name] = torch.load(paths['scaler'], map_location=DEVICE, weights_only=False)
    except FileNotFoundError:
        print(f"🔴 ERROR: {model_name} 모델 또는 스케일러 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"🔴 ERROR: {model_name} 모델 로딩 중 오류 발생: {e}")
print("✅ 주가 예측 모델 로딩 완료")

# 4. 주가 데이터 로드
try:
    print("... 주가 데이터 로딩 중 ...")
    stock_df = pd.read_csv(STOCK_CSV_PATH, index_col='Date', parse_dates=True)
    stock_df.sort_index(inplace=True)
    print("✅ 주가 데이터 로딩 완료")
except FileNotFoundError:
    print(f"🔴 ERROR: 주가 데이터 파일을 찾을 수 없습니다: {STOCK_CSV_PATH}")
    stock_df = None
