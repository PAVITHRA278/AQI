import numpy as np
import pandas as pd
import pymongo
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from tensorflow.keras.losses import MeanSquaredError

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["AirQualityDB"]
collection = db["real_time_aqi"]

# Load trained models
lstm_model = load_model("lstm_model.h5", custom_objects={"mse": MeanSquaredError()})
rf_model = joblib.load("random_forest_model.pkl")

# Fetch latest AQI data
def get_latest_aqi(city):
    latest_data = collection.find_one({"city": city}, sort=[("timestamp", -1)])
    if latest_data:
        return latest_data["aqi"]
    else:
        return None

# Predict AQI using LSTM
def predict_lstm(latest_aqi):
    scaler = MinMaxScaler(feature_range=(0, 1))
    latest_aqi_scaled = scaler.fit_transform(np.array(latest_aqi).reshape(-1, 1))
    input_sequence = np.array(latest_aqi_scaled[-10:]).reshape(1, 10, 1)
    predictions = []
    
    for _ in range(90):  # Predict for 90 days
        pred = lstm_model.predict(input_sequence)[0][0]
        predictions.append(pred)
        input_sequence = np.roll(input_sequence, -1)
        input_sequence[0, -1, 0] = pred
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Predict AQI using Random Forest
def predict_rf(latest_aqi):
    X_rf = np.array(latest_aqi[-10:]).reshape(1, -1)
    predictions = []
    for _ in range(90):  # Predict for 90 days
        pred = rf_model.predict(X_rf)[0]
        predictions.append(pred)
        X_rf = np.roll(X_rf, -1)
        X_rf[0, -1] = pred
    return np.array(predictions)

# Hybrid Prediction (Average of LSTM & RF)
def predict_hybrid(city):
    latest_aqi = get_latest_aqi(city)
    if latest_aqi is None:
        return []
    
    lstm_preds = predict_lstm([latest_aqi] * 10)  # Repeat latest AQI for LSTM input
    rf_preds = predict_rf([latest_aqi] * 10)      # Repeat latest AQI for RF input
    hybrid_preds = (lstm_preds + rf_preds) / 2  # Averaging both predictions
    
    # Generate Dates for the next 3 months
    dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 91)]
    
    # Return as structured data for Streamlit UI
    prediction_data = [[dates[i], lstm_preds[i], rf_preds[i], hybrid_preds[i]] for i in range(len(dates))]
    
    return prediction_data  # Returning list of lists

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        city = sys.argv[1]
        predictions = predict_hybrid(city)
        for row in predictions:
            print(row)  # Print in tabular format for testing
    else:
        print("Error: No city provided.")
