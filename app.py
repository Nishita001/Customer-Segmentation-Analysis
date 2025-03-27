import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model & scaler
@st.cache_resource
def load_model():
    model = joblib.load("rfm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")  # Load feature names
    return model, scaler, feature_names

# Function for customer segment prediction
def predict_customer_segment(model, scaler, feature_names):
    st.title("🎯 Predict Customer Segment")

    # Input fields
    recency = st.number_input("📅 Recency (Days Since Last Purchase):", min_value=0, max_value=365, value=30)
    frequency = st.number_input("🔄 Frequency (Number of Purchases):", min_value=0, max_value=100, value=5)
    monetary = st.number_input("💰 Monetary Value (Total Spend):", min_value=0, max_value=10000, value=500)

    if st.button("🚀 Predict Segment"):
        # Create DataFrame with correct feature names
        user_input = pd.DataFrame([[recency, frequency, monetary]], columns=feature_names)
        
        # Scale input before prediction
        user_input_scaled = scaler.transform(user_input)

        # Predict cluster
        prediction = model.predict(user_input_scaled)

        # Mapping cluster numbers to meaningful names
        segment_names = {0: "VIP Customers", 1: "Regular Customers", 2: "At-Risk Customers", 3: "Lost Customers"}
        segment = segment_names.get(prediction[0], "Unknown Segment")

        st.success(f"🟢 Predicted Customer Segment: **{segment}**")

# Main function
def main():
    model, scaler, feature_names = load_model()
    predict_customer_segment(model, scaler, feature_names)

if __name__ == "__main__":
    main()
