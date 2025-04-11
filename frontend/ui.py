import gradio as gr  # type: ignore
import requests  # type: ignore

def predict_diabetes(ecg_input):
    """Send ECG data to Flask API and return the result"""
    url = "http://127.0.0.1:5000/predict"

    # Convert input string to a list of floats
    try:
        ecg_values = [float(value.strip()) for value in ecg_input.split(",") if value.strip()]
        if not ecg_values:
            return "Error: Input cannot be empty. Please enter comma-separated numeric values."
    except ValueError:
        return "Error: Please enter valid comma-separated numeric values (e.g., 0.5, 1.2, 3.4)."

    # Send request to Flask API
    try:
        response = requests.post(url, json={"ecg_values": ecg_values})
        if response.status_code == 200:
            return response.json().get("prediction", "No prediction received")
        else:
            return f"Error: {response.json().get('error', 'Invalid response from server')}"
    except requests.exceptions.ConnectionError:
        return "Error: Unable to connect to Flask API. Is the server running?"
    except requests.exceptions.RequestException as e:
        return f"Error: An unexpected issue occurred: {e}"

# Gradio UI
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=gr.Textbox(label="Enter ECG Signal (comma-separated)"),
    outputs=gr.Textbox(label="Diabetes Prediction"),
    title="ECG-Based Diabetes Detector",
    description="Enter ECG Signal values (comma-separated) to predict if the person is diabetic.",
    allow_flagging="never"  # Disable flagging to keep UI clean
)

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)  # Ensure it runs on 7860
