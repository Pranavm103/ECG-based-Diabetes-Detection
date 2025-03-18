import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("backend/trained_model.h5")


sample_ecg = np.array([[0.82, 1.2, 100, 0.3, 0.8, 120, 0.02, 140]])


prediction = model.predict(sample_ecg)


result = "Diabetic" if prediction[0][0] > 0.5 else "Non-Diabetic"
print(f"Prediction: {result}")
