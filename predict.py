import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import tensorflow as tf

# === Input values ===
salinity = 0.04
temperature = 31.20
humidity = 84.7
time_of_day = 11.51

# Binary rule-based features
high_temp = int(temperature > 35)
high_humidity = int(humidity > 65)
high_salinity = int(salinity > 2.0)

# Load scaler and scale continuous features
scaler = joblib.load("soil_scaler.pkl")
input_cont = scaler.transform([[salinity, temperature, humidity, time_of_day]])

# Combine scaled + binary features
input_final = np.hstack((input_cont, [[high_temp, high_humidity, high_salinity]])).astype(np.float32)

# Load and run TFLite model
interpreter = tf.lite.Interpreter(model_path="soil_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_final)
interpreter.invoke()

# Get prediction
output = interpreter.get_tensor(output_details[0]['index'])
confidence = float(output[0][0])
predicted = int(confidence > 0.5)

print(f"\nConfidence: {confidence:.4f}")
print("💧 MOIST" if predicted == 0 else "❌ DRY")
