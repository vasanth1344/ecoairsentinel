import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Parameters
WINDOW_SIZE = 72  # Use past 72 hours to predict next 72 hours
FORECAST_HORIZON = 72  # 3 days ahead (hourly)

# Load node1 and node2
node1 = pd.read_csv("node1.csv")
node2 = pd.read_csv("node2.csv")

def preprocess_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.drop(columns=["id"], errors="ignore", inplace=True)
    df = df.ffill().bfill()
    return df

node1 = preprocess_data(node1)
node2 = preprocess_data(node2)

# Combine node1 and node2 into one DataFrame (you can also train them separately)
node1["source"] = 1
node2["source"] = 2
df = pd.concat([node1, node2]).sort_values("timestamp").reset_index(drop=True)

# Feature selection
features = ["temperature", "humidity", "pm25", "pm10", "source"]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])
joblib.dump(scaler, "lstm_scaler.save")

def create_sequences(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_horizon, 2:4])  # pm25 & pm10
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, WINDOW_SIZE, FORECAST_HORIZON)

# Define model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(WINDOW_SIZE, len(features))),
    LSTM(32, activation='relu'),
    Dense(FORECAST_HORIZON * 2)  # 72 hours * 2 pollutants
])

model.compile(optimizer='adam', loss='mse')

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, y.reshape((y.shape[0], y.shape[1] * 2)),  # Flatten y for Dense output
          epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save model
model.save("lstm_multi_output_model_3day.h5")
print("✅ 3-day forecast model and scaler saved.")
