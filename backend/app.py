from flask import Flask, request, jsonify
import serial
import torch
import torch.nn as nn  # type: ignore
import numpy as np
import os

app = Flask(__name__)

# LSTM Model Definition
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

# Model Initialization
input_size = 9  # Ensure this matches dataset features
model = LSTMModel(input_size)

checkpoint_path = "lstm_model.pth"

# Load the model with error handling
if os.path.exists(checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()
        print("✅ LSTM Model Loaded Successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print(f"❌ Model file not found: {checkpoint_path}")
    model = None  # Prevent using an uninitialized model

# Serial Connection Setup
ser = None
try:
    ser = serial.Serial('COM5', 115200, timeout=1)
    print("✅ Serial port connected!")
except serial.SerialException as e:
    print(f"❌ Serial Port Error: {e}")

def preprocess_data(ecg_values):
    """ Convert raw ECG values to PyTorch tensor (reshaped for LSTM) """
    return torch.tensor([ecg_values], dtype=torch.float32).unsqueeze(0)

def get_ecg_from_serial():
    """ Read 9 ECG values from the serial port """
    if ser:
        try:
            raw_data = ser.readline().decode().strip()
            ecg_values = list(map(float, raw_data.split(",")))  # Convert to float list
            if len(ecg_values) == input_size:
                return ecg_values
            else:
                print(f"❌ Invalid ECG Data Length: {len(ecg_values)} instead of {input_size}")
        except Exception as e:
            print(f"❌ Serial Read Error: {e}")
    return None

@app.route("/predict", methods=["POST"])
def predict():
    """ Predict Diabetes using LSTM Model """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Try reading ECG values from Serial
    ecg_values = get_ecg_from_serial()

    # If Serial data is unavailable, get it from the UI request
    if not ecg_values:
        try:
            data = request.get_json()
            if not data or "ecg_values" not in data:
                return jsonify({"error": "Missing 'ecg_values' in request"}), 400
            ecg_values = data.get("ecg_values")
        except Exception as e:
            return jsonify({"error": f"Invalid JSON format: {e}"}), 400

    # Validate ECG input format
    if not isinstance(ecg_values, list) or len(ecg_values) != input_size:
        return jsonify({"error": f"Invalid ECG input! Expected list of {input_size} values."}), 400

    # Convert ECG values to tensor
    input_data = preprocess_data(ecg_values)

    # Make prediction
    with torch.no_grad():
        prediction = model(input_data)

    probability = prediction.item()
    result = "Diabetic" if probability > 0.5 else "Non-Diabetic"

    return jsonify({"prediction": result, "probability": probability})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        if ser:
            ser.close()  # Ensure the serial port is closed on exit
        print("✅ Server shutdown, serial port closed!")
