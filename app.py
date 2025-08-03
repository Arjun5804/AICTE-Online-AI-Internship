import streamlit as st
import numpy as np
import joblib
from datetime import datetime

model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="🔥 Fire Type Classifier",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("📘 About This App")
    st.markdown("""
    Predict the **type of fire** detected by MODIS satellite data from 2021–2023 using a machine learning model.

    - **Model**: Random Forest
    - **Trained on**: MODIS fire data (India)
    - **Target Classes**:
        - 🌿 Vegetation Fire
        - 🏜️ Static Land Source
        - 🌊 Offshore Fire
    - **Inputs**: 6 satellite parameters
    """)


st.title("🔥 MODIS Fire Type Classifier")
st.markdown("Enter MODIS satellite readings below to predict the type of fire detected.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        brightness = st.number_input("🔆 Brightness", value=310.0, help="Satellite-detected brightness value.")
        bright_t31 = st.number_input("🌡️ Brightness T31", value=295.0, help="Thermal infrared band (T31) brightness.")
        frp = st.number_input("🔥 Fire Radiative Power (FRP)", value=25.0, help="Energy radiated from the fire in MW.")

    with col2:
        scan = st.number_input("📡 Scan", value=1.0, help="Width of the scanned area.")
        track = st.number_input("🛰️ Track", value=1.0, help="Height of the scanned area.")
        confidence = st.selectbox("🎯 Confidence Level", ["low", "nominal", "high"], help="Confidence level of detection.")

    submitted = st.form_submit_button("🚨 Predict Fire Type")

if submitted:
    confidence_map = {"low": 0, "nominal": 1, "high": 2}
    confidence_val = confidence_map.get(confidence.lower(), 1)

    input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]

    fire_types = {
        0: "🌿 Vegetation Fire",
        2: "🏜️ Other Static Land Source",
        3: "🌊 Offshore Fire"
    }

    result = fire_types.get(prediction, "❓ Unknown")

    st.markdown("### ✅ Prediction Result")
    st.success(f"**Predicted Fire Type:** {result}")
    st.caption(f"🕒 Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
