import serial  
import json

ser = serial.Serial("COM5", 115200, timeout=1)  # Change COM5 to your port

while True:
    try:
        line = ser.readline().decode("utf-8", errors="ignore").strip()  # Ignore decoding errors
        if line and line.startswith("{"):
            ecg_data = json.loads(line)  # Convert JSON string to dict
            ecg_values = ecg_data["ecg_values"]

            avg_ecg = sum(ecg_values) / len(ecg_values)  # Compute average ECG
            status = "Diabetic ⚠️" if avg_ecg > 0.5 else "Non-Diabetic ✅"

            print(f"ECG: {ecg_values}")
            print(f"Average: {avg_ecg:.3f} mV --> {status}\n")

    except json.JSONDecodeError:
        print("❌ JSON Decode Error: Invalid Data Received")
    except Exception as e:
        print("❌ Error:", e)
