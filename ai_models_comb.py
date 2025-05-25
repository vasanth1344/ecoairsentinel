# ai_models_combined.py
import os

# Ensure the static folder exists
os.makedirs("static", exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load data
node1 = pd.read_csv("node1.csv")
node2 = pd.read_csv("node2.csv")

# Align timestamps
common_timestamps = set(node1["timestamp"]) & set(node2["timestamp"])
node1 = node1[node1["timestamp"].isin(common_timestamps)].sort_values("timestamp").reset_index(drop=True)
node2 = node2[node2["timestamp"].isin(common_timestamps)].sort_values("timestamp").reset_index(drop=True)

# ---------- LSTM FORECAST (using both nodes) ----------
model_lstm = load_model("lstm_multi_output_model.h5")
scaler = joblib.load("lstm_scaler.save")

recent_data1 = node1[-20:][["temperature", "humidity", "pm25", "pm10"]].values
recent_data2 = node2[-20:][["temperature", "humidity", "pm25", "pm10"]].values
scaled_recent1 = scaler.transform(recent_data1)
scaled_recent2 = scaler.transform(recent_data2)
X_pred1 = np.expand_dims(scaled_recent1, axis=0)
X_pred2 = np.expand_dims(scaled_recent2, axis=0)
pred_scaled1 = model_lstm.predict(X_pred1)[0]
pred_scaled2 = model_lstm.predict(X_pred2)[0]
pred_full1 = np.concatenate([np.zeros(2), pred_scaled1])
pred_full2 = np.concatenate([np.zeros(2), pred_scaled2])
pred_inverse1 = scaler.inverse_transform([pred_full1])[0][2:]
pred_inverse2 = scaler.inverse_transform([pred_full2])[0][2:]

forecast_pm25_1, forecast_pm10_1 = pred_inverse1
forecast_pm25_2, forecast_pm10_2 = pred_inverse2

plt.figure(figsize=(8, 4))
x_labels = ["Node1 PM2.5", "Node1 PM10", "Node2 PM2.5", "Node2 PM10"]
y_values = [forecast_pm25_1, forecast_pm10_1, forecast_pm25_2, forecast_pm10_2]
colors = ["#ff9999", "#66b3ff", "#ffcc99", "#99ff99"]
plt.bar(x_labels, y_values, color=colors)
plt.title("Forecasted PM2.5 and PM10 for Both Nodes")
plt.ylabel("Concentration (ug/m3)")
plt.tight_layout()
plt.savefig("static/forecast.png")

# ---------- XGBOOST & SHAP ANALYSIS (using both nodes) ----------
def train_shap_model(node_df, node_label):
    X = node_df[["temperature", "humidity"]]
    y = node_df[["pm25", "pm10"]]
    model = MultiOutputRegressor(xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4))
    model.fit(X, y)
    explainer = shap.Explainer(model.estimators_[0])
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Explanation for PM2.5 ({node_label})")
    plt.tight_layout()
    plt.savefig(f"static/shap_pm25_{node_label.lower()}.png")

train_shap_model(node1, "Node1")
train_shap_model(node2, "Node2")

# ---------- NODE COMPARISON MODEL ----------
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
y_pred = model_cmp.predict(X_test)

comparison = pd.DataFrame({
    "PM2.5 Node1": node1["pm25"],
    "PM2.5 Node2": node2["pm25"],
    "PM10 Node1": node1["pm10"],
    "PM10 Node2": node2["pm10"]
})
plt.figure(figsize=(10, 5))
sns.lineplot(data=comparison["PM2.5 Node1"], label="Node1 PM2.5")
sns.lineplot(data=comparison["PM2.5 Node2"], label="Node2 PM2.5")
plt.title("PM2.5 Comparison between Nodes")
plt.savefig("static/compare_pm25.png")

plt.figure(figsize=(10, 5))
sns.lineplot(data=comparison["PM10 Node1"], label="Node1 PM10")
sns.lineplot(data=comparison["PM10 Node2"], label="Node2 PM10")
plt.title("PM10 Comparison between Nodes")
plt.savefig("static/compare_pm10.png")

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
