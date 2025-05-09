#  Air Quality Prediction System (Hybrid LSTM + Random Forest)

This project predicts Air Quality Index (AQI) levels using a **hybrid machine learning model** combining **Long Short-Term Memory (LSTM)** networks and **Random Forest (RF)** regressors. It leverages real-time AQI data fetched from the **AQICN API**, processes it, stores it in **MongoDB**, and trains predictive models for accurate future AQI forecasting and classification.

---
# AI-Powered Air Pollution Forecasting System

## Overview
This project develops an AI-powered system to forecast air quality using real-time data from the WAQI API, a hybrid Random Forest (RF) and Long Short-Term Memory (LSTM) model, and an interactive Streamlit web interface. It addresses limitations in existing systems by providing real-time integration, advanced predictions, and localized alerts for cities like New Delhi and Mumbai. Data is stored and managed using MongoDB.

## Features
- Fetches real-time AQI data from the WAQI API.
- Preprocesses data (handles missing values, removes outliers, normalizes) and stores it in MongoDB.
- Trains a hybrid RF+LSTM model for accurate 180-day AQI forecasts.
- Displays predictions and alerts via a user-friendly Streamlit interface.
- Supports city-specific forecasting with potential for future expansion.

## Prerequisites
- **Python 3.8+**
- **MongoDB**: Ensure it is installed and running (`mongod`).
- **Git** (optional, for version control).

## Installation

1. **Clone the Repository** (if using Git):
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

---


---

## ⚙️ How It Works

### 1️⃣ Preprocess AQI Data  
**(preprocess_aqi(city))**
- Fetches real-time data via API.
- Cleans missing values, handles outliers (AQI < 500), and normalizes using **MinMaxScaler**.
- Stores preprocessed data in MongoDB.

### 2️⃣ Train Hybrid Models  
**(train_hybrid_models())**
- Loads preprocessed data from MongoDB.
- Prepares **time-series sequences** for LSTM and structured data for RF.
- Trains:
  - `RandomForestRegressor`
  - `Sequential LSTM` model
- Combines predictions using a **weighted hybrid (e.g. 60% LSTM, 40% RF)**.

### 3️⃣ Evaluate and Predict  
- Classifies AQI into categories like *Good, Moderate, Unhealthy, etc.*
- Displays metrics (Accuracy: **0.96**, RMSE: **10.0**, MAE: **8.5**).


**Install them using:**
```bash
pip install -r requirements.txt

****How to Run******
python app.py

https://22d5-2409-40f4-40c8-ced6-84bd-f1f6-c3fa-c67e.ngrok-free.app/

