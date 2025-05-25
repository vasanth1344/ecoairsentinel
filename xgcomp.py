import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load data
node1 = pd.read_csv("node1.csv")
node2 = pd.read_csv("node2.csv")

# Align timestamps
common_timestamps = set(node1["timestamp"]) & set(node2["timestamp"])
node1 = node1[node1["timestamp"].isin(common_timestamps)].sort_values("timestamp").reset_index(drop=True)
node2 = node2[node2["timestamp"].isin(common_timestamps)].sort_values("timestamp").reset_index(drop=True)

# Features and targets
X = pd.DataFrame({
    "temperature_node1": node1["temperature"],
    "humidity_node1": node1["humidity"],
    "temperature_node2": node2["temperature"],
    "humidity_node2": node2["humidity"]
})
y = pd.DataFrame({
    "pm25_diff": node1["pm25"] - node2["pm25"],
    "pm10_diff": node1["pm10"] - node2["pm10"]
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
base_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae_pm25 = mean_absolute_error(y_test["pm25_diff"], y_pred[:, 0])
mae_pm10 = mean_absolute_error(y_test["pm10_diff"], y_pred[:, 1])

# --- VISUALIZATION ---

# Scatter plots for prediction
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test["pm25_diff"], y_pred[:, 0], alpha=0.6, color="skyblue")
plt.plot([y_test["pm25_diff"].min(), y_test["pm25_diff"].max()],
         [y_test["pm25_diff"].min(), y_test["pm25_diff"].max()], "k--")
plt.title(f"PM2.5 Difference\nMAE: {mae_pm25:.2f}")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test["pm10_diff"], y_pred[:, 1], alpha=0.6, color="salmon")
plt.plot([y_test["pm10_diff"].min(), y_test["pm10_diff"].max()],
         [y_test["pm10_diff"].min(), y_test["pm10_diff"].max()], "k--")
plt.title(f"PM10 Difference\nMAE: {mae_pm10:.2f}")
plt.xlabel("Actual")
plt.ylabel("Predicted")

plt.tight_layout()
plt.savefig("pm_prediction_scatter.png")
plt.show()

# Pie charts for PM contribution
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
plt.savefig("pm_contribution_pie.png")
plt.show()
