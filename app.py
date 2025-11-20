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

st.set_page_config(page_title="AAPL Stock Prediction RNN", layout="wide")
st.title("📈 AAPL Stock Price Prediction (RNN)")

st.markdown("""
### How to Use:
Choose an input method below to generate stock price data and see predictions!
- **Random Data**: Generate realistic random stock prices
- **Custom Range**: Enter a price range to generate data
- **Manual Input**: Paste your own prices (comma-separated)
- **Upload CSV**: Upload a CSV file with historical data
""")

# Sidebar for input method selection
input_method = st.radio("Select input method:", ["Random Data", "Custom Range", "Manual Input", "Upload CSV"])

seq_length = 60
data = None

if input_method == "Random Data":
    st.subheader("🎲 Generate Random Stock Prices")
    
    col1, col2 = st.columns(2)
    with col1:
        num_days = st.slider("Number of days to generate:", min_value=100, max_value=500, value=200, step=10)
        base_price = st.slider("Base stock price ($):", min_value=50.0, max_value=300.0, value=150.0, step=5.0)
    with col2:
        volatility = st.slider("Price volatility (%):", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        trend = st.slider("Trend direction:", min_value=-0.5, max_value=0.5, value=0.0, step=0.1)
    
    if st.button("Generate & Predict", key="random"):
        # Generate realistic random walk data
        prices = [base_price]
        for _ in range(num_days - 1):
            change = np.random.normal(trend, volatility)
            new_price = max(prices[-1] * (1 + change / 100), 10)  # Ensure price stays positive
            prices.append(new_price)
        
        data = np.array(prices).reshape(-1, 1)
        st.success(f"✓ Generated {num_days} days of stock prices")

elif input_method == "Custom Range":
    st.subheader("📊 Generate Prices in Custom Range")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_price = st.number_input("Minimum price ($):", min_value=1.0, max_value=1000.0, value=100.0)
    with col2:
        max_price = st.number_input("Maximum price ($):", min_value=1.0, max_value=1000.0, value=200.0)
    with col3:
        num_days = st.number_input("Number of days:", min_value=100, max_value=500, value=200, step=10)
    
    if st.button("Generate & Predict", key="custom"):
        if min_price >= max_price:
            st.error("Minimum price must be less than maximum price!")
        else:
            # Generate smooth transition between min and max
            prices = np.linspace(min_price, max_price, num_days)
            noise = np.random.normal(0, (max_price - min_price) * 0.02, num_days)
            prices = prices + noise
            prices = np.clip(prices, min_price, max_price)
            
            data = prices.reshape(-1, 1)
            st.success(f"✓ Generated {num_days} days of stock prices")

elif input_method == "Manual Input":
    st.subheader("✏️ Enter Stock Prices Manually")
    price_input = st.text_area("Enter prices separated by commas (e.g., 150.5, 152.3, 151.8, ...):", 
                               placeholder="150, 151, 152, 153, 154")
    
    if st.button("Predict", key="manual"):
        try:
            prices = [float(p.strip()) for p in price_input.split(",") if p.strip()]
            if len(prices) < seq_length:
                st.error(f"⚠️ Please enter at least {seq_length} prices!")
            else:
                data = np.array(prices).reshape(-1, 1)
                st.success(f"✓ Loaded {len(prices)} prices")
        except ValueError:
            st.error("❌ Invalid format! Make sure prices are separated by commas.")

elif input_method == "Upload CSV":
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with AAPL prices:", type=["csv"])
    
    if uploaded_file and st.button("Predict", key="upload"):
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())
        
        if "Adj Close" not in df.columns:
            st.error("❌ CSV must contain 'Adj Close' column!")
        else:
            data = df["Adj Close"].values.reshape(-1, 1)
            st.success(f"✓ Loaded {len(data)} prices from CSV")

# Make predictions if data is available
if data is not None:
    try:
        # Preprocess input data
        scaled = scaler.transform(data)
        
        # Make sequences (60-day window)
        X = []
        for i in range(seq_length, len(scaled)):
            X.append(scaled[i-seq_length:i])
        
        if len(X) == 0:
            st.error(f"⚠️ Not enough data points. Need at least {seq_length + 1} prices!")
        else:
            X = np.array(X)
            
            # Predict
            with st.spinner("🔮 Making predictions..."):
                predictions = model.predict(X, verbose=0)
            predictions = scaler.inverse_transform(predictions)
            
            # Crop actual to match prediction length
            actual = data[seq_length:]
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Actual Price", f"${actual.mean():.2f}")
            with col2:
                st.metric("Average Predicted Price", f"${predictions.mean():.2f}")
            with col3:
                mape = np.mean(np.abs((actual - predictions) / actual)) * 100
                st.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")
            
            # Plot
            st.subheader("📈 Prediction Results")
            fig, ax = plt.subplots(figsize=(14, 6))
            
            days = np.arange(len(actual))
            ax.plot(days, actual, label="Actual Price", linewidth=2.5, marker="o", markersize=4, color="#1f77b4")
            ax.plot(days, predictions, label="Predicted Price", linewidth=2.5, marker="s", markersize=4, color="#ff7f0e")
            
            ax.fill_between(days, actual.flatten(), predictions.flatten(), alpha=0.2, color="gray")
            ax.legend(fontsize=12, loc="best")
            ax.set_xlabel("Days", fontsize=11)
            ax.set_ylabel("Stock Price ($)", fontsize=11)
            ax.set_title("AAPL Stock Price: Actual vs Predicted (RNN Model)", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Display data table
            st.subheader("📊 Detailed Results")
            results_df = pd.DataFrame({
                "Day": days,
                "Actual Price ($)": actual.flatten(),
                "Predicted Price ($)": predictions.flatten(),
                "Difference ($)": (actual - predictions).flatten(),
                "Error (%)": ((actual - predictions) / actual * 100).flatten()
            })
            st.dataframe(results_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

st.markdown("---")
st.markdown("""
**About this RNN Model:**
- Uses LSTM (Long Short-Term Memory) neural network
- Trained on AAPL historical stock prices
- Uses a 60-day window to predict the next day's price
- Normalizes data using Min-Max scaler for better performance
""")