import pandas as pd
import numpy as np
import pymongo
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# 🔹 Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["AirQualityDB"]
collection = db["real_time_aqi"]

# 🔹 Fetch AQI data from MongoDB
data = list(collection.find({}, {"_id": 0, "city": 1, "timestamp": 1, "aqi": 1}))
df = pd.DataFrame(data)

# 🔹 Ensure Data Exists
if df.empty or df.shape[0] < 10:
    raise ValueError("❌ Not enough AQI data available for training. At least 10 records required.")

# 🔹 Convert 'timestamp' to datetime & Sort
df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df = df.dropna().sort_values(by="timestamp")

# 🔹 Normalize AQI Data
scaler = MinMaxScaler()
df["aqi"] = scaler.fit_transform(df[["aqi"]])

# 🔹 Save the Scaler for Future Use (Important)
joblib.dump(scaler, "scaler.pkl")

# 🔹 Prepare Training Data
X, y = [], []
sequence_length = 10  # Use last 10 AQI values to predict the next one

for i in range(len(df) - sequence_length):
    X.append(df["aqi"].values[i:i+sequence_length])
    y.append(df["aqi"].values[i+sequence_length])

X, y = np.array(X), np.array(y)

# 🔹 Ensure X shape compatibility (no extra dimensions needed for RF)
X = X.reshape(X.shape[0], X.shape[1])  

# 🔹 Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train Random Forest Model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# 🔹 Save the Trained Model
joblib.dump(model_rf, "random_forest_model.pkl")

print("✅ Random Forest Model Trained & Saved Successfully!")
