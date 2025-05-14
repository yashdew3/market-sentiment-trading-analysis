import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Load model and scaler
model = joblib.load("outputs/models/sentiment_rf_model_optimized.pkl")
scaler = joblib.load("outputs/models/scaler.pkl")

# Page configuration
st.set_page_config(page_title="📊 Market Sentiment Dashboard", layout="wide")

# Title
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📈 Market Sentiment Analysis Dashboard</h1>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📤 Upload or Simulate Trade Data")
    st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

sample_data_path = "data/feature_engineered_data.csv"
df = pd.read_csv(sample_data_path)
df['date'] = pd.to_datetime(df['date'])

# Show EDA Charts
st.subheader("🔍 Historical Trends by Sentiment")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df, x='date', y='Total_PnL', hue='classification', ax=ax)
    ax.set_title("Total PnL Over Time", fontsize=14, color="#4CAF50")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total PnL")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df, x='date', y='Total_Trade_Volume', hue='classification', ax=ax2)
    ax2.set_title("Trade Volume Over Time", fontsize=14, color="#4CAF50")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Trade Volume")
    ax2.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig2)

# Upload CSV or use sample input
st.subheader("🧪 Predict Sentiment from Trade Activity")
st.markdown("<p style='color: gray;'>Upload your own data or use sample trades to see predicted sentiment labels.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV with trade activity (or use sample below)", type="csv")

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
else:
    st.info("Using 5 recent rows from sample data")
    input_df = df.dropna().sample(5, random_state=42)

# Feature columns used in model
features = [
    'Total_PnL', 'Total_Trade_Volume', 'Trade_Count',
    'PnL_3D_Mean', 'PnL_3D_STD',
    'Volume_3D_Mean', 'TradeCount_3D_Mean'
]

if all(col in input_df.columns for col in features):
    scaled_input = scaler.transform(input_df[features])
    predictions = model.predict(scaled_input)
    input_df['Predicted Sentiment'] = predictions
    input_df['Predicted Sentiment'] = input_df['Predicted Sentiment'].map({0: 'Fear', 1: 'Greed'})

    st.markdown("""
        <style>
        .dataframe th, .dataframe td {
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    st.dataframe(input_df[['date'] + features + ['Predicted Sentiment']], use_container_width=True)
else:
    st.warning("❌ Uploaded data missing required feature columns.")

st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p style='color: #777;'>Built with ❤️ using <strong>Streamlit</strong>, <strong>Scikit-learn</strong>, and <strong>Seaborn</strong></p>
    </div>
""", unsafe_allow_html=True)
