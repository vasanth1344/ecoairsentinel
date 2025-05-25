import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Load data from database
def load_data_from_db():
    conn = sqlite3.connect("new.db")
    node1 = pd.read_sql_query("SELECT * FROM node1_sensor_readings", conn)
    node2 = pd.read_sql_query("SELECT * FROM node2_sensor_readings", conn)
    conn.close()
    return node1, node2

node1, node2 = load_data_from_db()

# Align timestamps
common_timestamps = set(node1["timestamp"]) & set(node2["timestamp"])
node1 = node1[node1["timestamp"].isin(common_timestamps)].sort_values("timestamp").reset_index(drop=True)
node2 = node2[node2["timestamp"].isin(common_timestamps)].sort_values("timestamp").reset_index(drop=True)

# ---------- LSTM FORECAST FOR 3 DAYS ----------
model_lstm = load_model("lstm_multi_output_model_3day.h5", compile=False)
scaler = joblib.load("lstm_scaler.save")

def forecast_3days(node_data, source_value):
    recent_data = node_data[["temperature", "humidity", "pm25", "pm10"]].values[-20:]
    source_col = np.full((recent_data.shape[0], 1), source_value)
    full_data = np.hstack([recent_data, source_col])
    scaled_recent = scaler.transform(full_data)
    X_pred = np.expand_dims(scaled_recent, axis=0)
    pred_scaled = model_lstm.predict(X_pred)[0]
    pad = np.zeros((1, 3))  # temperature, humidity, source
    pred_padded = np.hstack([pad.repeat(72, axis=0), pred_scaled.reshape(72, 2)])
    pred_inverse = scaler.inverse_transform(pred_padded)[:, -2:]
    return pred_inverse

forecast_node1 = forecast_3days(node1, source_value=0)
forecast_node2 = forecast_3days(node2, source_value=1)

# Plot forecast
plt.figure(figsize=(12, 5))
plt.plot(forecast_node1[:, 0], label="Node1 PM2.5", color="#ff9999")
plt.plot(forecast_node1[:, 1], label="Node1 PM10", color="#66b3ff")
plt.plot(forecast_node2[:, 0], label="Node2 PM2.5", color="#ffcc99")
plt.plot(forecast_node2[:, 1], label="Node2 PM10", color="#99ff99")
plt.title("3-Day Forecast of PM2.5 and PM10")
plt.xlabel("Hour Ahead")
plt.ylabel("Concentration (µg/m³)")
plt.legend()
plt.tight_layout()
plt.savefig("static/forecast_3day.png")
plt.close()

# ---------- SHAP ANALYSIS ----------
def train_shap_model(node_df, node_label):
    features = ["temperature", "humidity"]
    for target in ["pm25", "pm10"]:
        X = node_df[features]
        y = node_df[target]
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4)
        model.fit(X, y)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Summary for {target.upper()} ({node_label})")
        plt.tight_layout()
        plt.savefig(f"static/shap_summary_{target}_{node_label.lower()}.png")
        plt.close()

        for feature in features:
            shap.dependence_plot(feature, shap_values.values, X, show=False, interaction_index=None)
            plt.title(f"{target.upper()} vs {feature.capitalize()} ({node_label})")
            plt.tight_layout()
            plt.savefig(f"static/shap_depend_{target}_{feature}_{node_label.lower()}.png")
            plt.close()

train_shap_model(node1, "Node1")
train_shap_model(node2, "Node2")

# ---------- NODE COMPARISON ----------
X_diff = pd.DataFrame({
    "temperature_node1": node1["temperature"],
    "humidity_node1": node1["humidity"],
    "temperature_node2": node2["temperature"],
    "humidity_node2": node2["humidity"]
})
y_diff = pd.DataFrame({
    "pm25_diff": node1["pm25"] - node2["pm25"],
    "pm10_diff": node1["pm10"] - node2["pm10"]
})
X_train, X_test, y_train, y_test = train_test_split(X_diff, y_diff, test_size=0.2, random_state=42)
model_cmp = MultiOutputRegressor(xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4))
model_cmp.fit(X_train, y_train)

# PM comparison line plots
comparison = pd.DataFrame({
    "PM2.5 Node1": node1["pm25"],
    "PM2.5 Node2": node2["pm25"],
    "PM10 Node1": node1["pm10"],
    "PM10 Node2": node2["pm10"]
})
plt.figure(figsize=(10, 5))
sns.lineplot(data=comparison[["PM2.5 Node1", "PM2.5 Node2"]])
plt.title("PM2.5 Comparison between Nodes")
plt.savefig("static/compare_pm25.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.lineplot(data=comparison[["PM10 Node1", "PM10 Node2"]])
plt.title("PM10 Comparison between Nodes")
plt.savefig("static/compare_pm10.png")
plt.close()

# ---------- PIE CHART ----------
total_pm25_node1 = node1["pm25"].sum()
total_pm25_node2 = node2["pm25"].sum()
total_pm10_node1 = node1["pm10"].sum()
total_pm10_node2 = node2["pm10"].sum()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie([total_pm25_node1, total_pm25_node2], labels=["Node1", "Node2"],
        autopct='%1.1f%%', colors=["#66b3ff", "#ff9999"], startangle=140)
plt.title("PM2.5 Contribution")

plt.subplot(1, 2, 2)
plt.pie([total_pm10_node1, total_pm10_node2], labels=["Node1", "Node2"],
        autopct='%1.1f%%', colors=["#99ff99", "#ffcc99"], startangle=140)
plt.title("PM10 Contribution")

plt.tight_layout()
plt.savefig("static/contribution_pie.png")
plt.close()
