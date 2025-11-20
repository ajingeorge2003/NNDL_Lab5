import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("stock_rnn.keras")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("📈 AAPL Stock Price Prediction (RNN)")

uploaded_file = st.file_uploader("Upload a CSV file with AAPL prices:", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Preprocess input data (expects 'Adj Close' column)
    data = df["Adj Close"].values.reshape(-1, 1)
    scaled = scaler.transform(data)

    # Make sequences (same as training: 60-day window)
    seq_length = 60
    X = []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i-seq_length:i])
    X = np.array(X)

    # Predict
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Crop actual to match prediction length
    actual = data[seq_length:]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual, label="Actual Price", linewidth=2)
    ax.plot(predictions, label="Predicted Price", linewidth=2)
    ax.legend()
    ax.set_title("AAPL Stock Prediction")
    st.pyplot(fig)