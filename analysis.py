from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
df=pd.read_csv("node1.csv")
# Load model and scaler
model = load_model("lstm_multi_output_model.h5", compile=False)  # <-- Fix: Avoid deserializing 'mse'
scaler = joblib.load("lstm_scaler.save")

# Assuming `df` is already defined, if not, load it from file
# Example: df = pd.read_csv("node1.csv")

# Prepare your latest data window (20 most recent rows)
recent_data = df[-20:][["temperature", "humidity", "pm25", "pm10"]].values
scaled_recent = scaler.transform(recent_data)
X_pred = np.expand_dims(scaled_recent, axis=0)  # shape (1, 20, 4)

# Predict
pred_scaled = model.predict(X_pred)[0]  # shape (2,)
pred_full = np.concatenate([np.zeros(2), pred_scaled])  # dummy to fill full feature vector
pred_inverse = scaler.inverse_transform([pred_full])[0][2:]  # unscale only pm25, pm10

print("Forecasted PM2.5:", pred_inverse[0])
print("Forecasted PM10:", pred_inverse[1])
