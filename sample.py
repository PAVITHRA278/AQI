


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pymongo
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["AirQualityDB"]
collection = db["real_time_aqi"]

import keras

@keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return keras.losses.MeanSquaredError()(y_true, y_pred)

# Load Models
lstm_model = load_model("lstm_model.h5", custom_objects={"mse": mse})
rf_model = joblib.load("random_forest_model.pkl")

# **Login System**
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login to Air Quality Prediction System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "password123":
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Invalid credentials. Please try again.")
else:
    # Streamlit UI
    st.title("Smart Air Quality Prediction System")
    st.markdown("📌 **Real-Time & Predicted AQI Data**")

    city_name = st.text_input("Enter City Name")
    
    

    def get_past_aqi(city, days=20):
        """Fetch the past days AQI values for the given city from MongoDB."""
        past_entries = collection.find({"city": city}).sort("timestamp", -1).limit(days)
        aqi_values = [entry.get("aqi", None) for entry in past_entries]

        # If not enough data is available, pad with average
        if len(aqi_values) < days:
            avg_aqi = np.mean(aqi_values) if aqi_values else 50  # Default average
            missing_values = [avg_aqi] * (days - len(aqi_values))
            aqi_values = missing_values + aqi_values

        return np.array(aqi_values).reshape(-1, 1)
  
    def get_latest_aqi(city):
        latest_entry = collection.find_one({"city": city}, sort=[("timestamp", -1)])
        if latest_entry:
            latest_entry["timestamp"] = latest_entry["timestamp"].to_pydatetime() if isinstance(latest_entry["timestamp"], pd.Timestamp) else latest_entry["timestamp"]
        return latest_entry

    
    

    # **REAL-TIME AQI DISPLAY**
    if st.button("Real-time AQI"):
        if city_name:
            latest_entry = get_latest_aqi(city_name)
            if latest_entry:
                aqi_values = {
                    "AQI": latest_entry.get("aqi", "N/A"),
                    "PM2.5": latest_entry.get("pm25", "N/A"),
                    "PM10": latest_entry.get("pm10", "N/A"),
                    "NO2": latest_entry.get("no2", "N/A"),
                    "CO": latest_entry.get("co", "N/A"),
                    "O3": latest_entry.get("o3", "N/A"),
                    "SO2": latest_entry.get("so2", "N/A"),
                    "Date": latest_entry.get("timestamp", "N/A"),
                }
                st.table(pd.DataFrame([aqi_values]))
            else:
                st.warning(f"No real-time AQI data found for {city_name}.")
        else:
            st.warning("Please enter a city name before fetching AQI data.")


    def predict_lstm(past_aqi):
        """Predict future AQI using LSTM model based on past AQI values."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        past_aqi_scaled = scaler.fit_transform(past_aqi)

        # Reshape for LSTM input
        input_seq = past_aqi_scaled.reshape(1, len(past_aqi), 1)

        predictions = []
        for _ in range(180):  # Predict next 6 months
            pred_scaled = lstm_model.predict(input_seq)[0][0]
            predictions.append(pred_scaled)
            input_seq = np.roll(input_seq, -1)
            input_seq[0, -1, 0] = pred_scaled

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return np.clip(predictions, 0, 500)

    def predict_rf(past_aqi):
          scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Use only the last 10 days for prediction
          past_aqi = past_aqi[-10:]  # Take the last 10 values to match training data
    
          past_aqi_scaled = scaler.fit_transform(past_aqi)
          X_rf = past_aqi_scaled.reshape(1, -1)  # Reshape for prediction
    
          predictions = []
          for _ in range(180): 
                pred_scaled = rf_model.predict(X_rf)[0]  # Predict next step
                pred = scaler.inverse_transform([[pred_scaled]])[0][0]  # Rescale back
                predictions.append(pred)

        # Shift values in input for rolling prediction
                X_rf = np.roll(X_rf, -1)
                X_rf[0, -1] = pred_scaled
    
          return np.array(predictions)
    def predict_hybrid(city):
        past_aqi = get_past_aqi(city, days=20)
        if past_aqi is None:
            return None

        lstm_preds = predict_lstm(past_aqi)
        rf_preds = predict_rf(past_aqi)
        hybrid_preds = 0.6 * lstm_preds + 0.4 * rf_preds
        
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 181)]
        df = pd.DataFrame({"Date": dates, "Hybrid": hybrid_preds})

        # Group by Month and Fix Ordering
        df["Month"] = pd.to_datetime(df["Date"]).dt.strftime("%B")
        df = df.groupby("Month")["Hybrid"].mean().reset_index()
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        df["Month"] = pd.Categorical(df["Month"], categories=month_order, ordered=True)
        #df = df.sort_values("Month")
        df = df.sort_values(by="Month").reset_index(drop=True)


        return df

    if st.button("Predict Future AQI"):
        if city_name:
            predictions = predict_hybrid(city_name)
            if predictions is not None:
                st.write("### Monthly Average AQI Predictions")
                st.table(predictions)
                st.session_state["df_pred"] = predictions
            else:
                st.warning("No data available for the entered city.")
        else:
            st.warning("Please enter a city name before predicting AQI.")

    if "df_pred" in st.session_state and st.button("Show Graph"):
        df_pred = st.session_state["df_pred"]
        plt.figure(figsize=(10, 5))

        # Fetch Real-time AQI for Graph
        real_aqi = get_past_aqi(city_name, days=1)[-1][0]

        plt.plot([0], [real_aqi], marker="^", linestyle="-.", color="blue", label="Real-time AQI")
        plt.plot(range(1, len(df_pred) + 1), df_pred["Hybrid"], marker="s", linestyle="--", color="green", label="Future Prediction")

        plt.xticks(range(0, len(df_pred) + 1), ["Real-Time"] + list(df_pred["Month"]), rotation=45)

        y_min = min(real_aqi, df_pred["Hybrid"].min())
        y_max = max(real_aqi, df_pred["Hybrid"].max())

        plt.ylim(y_min - 10, y_max + 10)
        plt.xlabel("Time")
        plt.ylabel("AQI Level")
        plt.title("Predicted AQI Levels for 6 Months")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)

# Logout Button
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun() 