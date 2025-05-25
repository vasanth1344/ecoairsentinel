import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import shap
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load your CSV data
df = pd.read_csv("feeds.csv")

# 2. Basic cleaning: Drop rows with missing values
df = df.dropna()

# 3. Select features and target
features = ['temperature', 'humidity', 'pm2_5', 'pm10']  # example
target = 'pm2_5'  # prediction target

X = df[features]
y = df[target]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train and save Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")

# 6. Train and save XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_model.save_model("xgb_model.json")

# 7. SHAP explainability
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary_plot.png")

# 8. Prepare data for LSTM
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

def create_lstm_dataset(data, time_steps=10):
    X_lstm, y_lstm = [], []
    for i in range(time_steps, len(data)):
        X_lstm.append(data[i-time_steps:i])
        y_lstm.append(data[i, 0])  # predicting 'temperature' or first feature
    return np.array(X_lstm), np.array(y_lstm)

X_lstm, y_lstm = create_lstm_dataset(scaled)

# Train/test split for LSTM
split_idx = int(len(X_lstm) * 0.8)
X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
y_lstm_train, y_lstm_test = y_lstm[:split_idx], y_lstm[split_idx:]

# 9. Build and train LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32,
               validation_split=0.1, callbacks=[EarlyStopping(patience=3)])

# 10. Save LSTM model
lstm_model.save("lstm_model.h5")

print("✅ All models trained and saved successfully.")
