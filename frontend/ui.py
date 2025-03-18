import gradio as gr
import requests


def predict_diabetes(ecg_signal):
    """ Send data to Flask API and return the result """
    url = "http://127.0.0.1:5000/predict"
    response = requests.post(url, json={"ecg_signal": ecg_signal})

    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return "Error: Invalid response"


# Gradio UI
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=gr.Number(label="Enter ECG Signal (mV)"),
    outputs=gr.Textbox(label="Diabetes Prediction"),
    title="ECG-Based Diabetes Detector",
    description="Enter ECG Signal to predict if the person is diabetic or not."
)

if __name__ == "__main__":
    interface.launch()
