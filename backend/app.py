from flask import Flask, request, jsonify
import serial
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])  # Take output from last time step
        return self.sigmoid(x)


input_size = 9  # Ensure this matches dataset features
model = LSTMModel(input_size)
checkpoint_path = "lstm_model.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
model.eval()

print("✅ LSTM Model Loaded Successfully!")


try:
    ser = serial.Serial('COM11', 115200, timeout=1)
    print("✅ Serial port connected!")
except serial.SerialException as e:
    print(f"❌ Serial Port Error: {e}")
    ser = None  # Prevent crashes


def preprocess_data(ecg_values):
    """ Converting raw ECG values to PyTorch tensor (reshaped for LSTM) """
    ecg_tensor = torch.tensor([ecg_values], dtype=torch.float32).unsqueeze(0)  # (batch, seq_len, features)
    return ecg_tensor

@app.route("/predict", methods=["POST"])
def predict():
    """ Predicting Diabetes using LSTM Model """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    ecg_values = data.get("ecg_values")

    if not ecg_values or len(ecg_values) != input_size:
        return jsonify({"error": "Invalid ECG input!"}), 400

    input_data = preprocess_data(ecg_values)

    with torch.no_grad():
        prediction = model(input_data)

    result = "Diabetic" if prediction.item() > 0.5 else "Non-Diabetic"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
