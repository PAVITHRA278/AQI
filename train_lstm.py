

def train_lstm():
    import numpy as np
    import pandas as pd
    import pymongo
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

# Connect to MongoDB
   # Connect to MongoDB
    client = pymongo.MongoClient("mongodb+srv://pavi270804:pavithra2708@cluster0.lmuuwot.mongodb.net/")
    db = client["AirQualityDB"]
    collection = db["real_time_aqi"]

# Fetch AQI data from MongoDB
    data = list(collection.find({}, {"_id": 0, "city": 1, "timestamp": 1, "aqi": 1}))

    if not data:
        print("⚠️ No AQI data found in MongoDB! Exiting...")
        exit()

# Convert to DataFrame
    df = pd.DataFrame(data)

# Ensure 'timestamp' is in datetime format and sort by time
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(str), errors='coerce')
    df = df.dropna(subset=["timestamp"])  # Remove invalid timestamps
    df = df.sort_values(by="timestamp")

# Drop missing AQI values
    df = df.dropna(subset=["aqi"])

# Normalize AQI data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["aqi"] = scaler.fit_transform(df[["aqi"]])

# Prepare training data
    sequence_length = 10  # Last 10 time steps for prediction
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df["aqi"].values[i:i + sequence_length])
        y.append(df["aqi"].values[i + sequence_length])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train & validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Build Improved LSTM Model
    model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

    model.compile(optimizer="adam", loss="mse", metrics=['mae'])

# Train Model
    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val))

# Save Model
    model.save("lstm_model.h5")
    print("✅ Model training complete! LSTM model saved as 'lstm_model.h5'.")
