import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Load data
node1 = pd.read_csv("node1.csv", parse_dates=["timestamp"])
features = ["temperature", "humidity", "pm10"]
target = "pm25"

node1_clean = node1.dropna(subset=features + [target])
X = node1_clean[features]
y = node1_clean[target]

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=4)
model.fit(X, y)

# SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Summary plot
shap.summary_plot(shap_values, X, show=True)

# Dependence plots
shap.dependence_plot("temperature", shap_values.values, X)
shap.dependence_plot("humidity", shap_values.values, X)
