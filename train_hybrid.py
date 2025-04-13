import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestRegressor

from sample import X_test, X_train,y_train

# Reshape for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(1)
])

lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)


# Save LSTM Model
lstm_model.save("lstm_model.h5")

# Predict using LSTM
y_pred_train_lstm = lstm_model.predict(X_train_lstm).flatten()
y_pred_test_lstm = lstm_model.predict(X_test_lstm).flatten()


# Train Hybrid Model (RF on LSTM output)
hybrid_model = RandomForestRegressor(n_estimators=100, random_state=42)
hybrid_model.fit(y_pred_train_lstm.reshape(-1, 1), y_train)

# Save Hybrid Model
joblib.dump(hybrid_model, "hybrid_model.pkl")

print("✅ Hybrid Model trained and saved successfully!")

