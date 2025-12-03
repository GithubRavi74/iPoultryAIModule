import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# -------------------------------------
# Load Models Safely
# -------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

weight_model = pickle.load(open(os.path.join(BASE_DIR, "weight_model.pkl"), "rb"))
mortality_model = pickle.load(open(os.path.join(BASE_DIR, "mortality_model.pkl"), "rb"))
fcr_model = pickle.load(open(os.path.join(BASE_DIR, "fcr_model.pkl"), "rb"))

# -------------------------------------
# Page Setup
# -------------------------------------
st.set_page_config(page_title="iPoultry AI Assistant", layout="wide")
st.title("🐔 iPoultry AI — Weight, Mortality & FCR Prediction")

st.markdown("### Enter today's flock metrics")

# -------------------------------------------------------
# Helper Function — Fake History for Demo (Replace Later)
# -------------------------------------------------------
def generate_demo_history(days=7):
    np.random.seed(42)
    data = {
        "temp": np.random.uniform(27, 30, days),
        "rh": np.random.uniform(55, 70, days),
        "co": np.random.uniform(300, 500, days),
        "nh3": np.random.uniform(10, 25, days),
        "feed": np.random.uniform(18, 25, days),
        "mortality": np.random.randint(0, 5, days),
        "weight": np.random.uniform(0.8, 1.5, days),
    }
    return pd.DataFrame(data)

history = generate_demo_history()

# -------------------------------------------------------
# Farmer Inputs
# -------------------------------------------------------
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age_in_days = st.number_input("Age in Days", 0, 100, 14)
        birds_alive = st.number_input("Birds Alive", 0, 200000, 900)
        mortality_today = st.number_input("Mortality Today", 0, 500, 1)

    with col2:
        feed_today = st.number_input("Feed Today (kg)", 0.0, 500.0, 22.0)
        water_today = st.number_input("Water Today (L)", 0.0, 800.0, 30.0)
        sample_weight_today = st.number_input("Sample Weight (kg)", 0.0, 5.0, 1.2)

    with col3:
        temp_today = st.number_input("Avg Temp Today (°C)", 0.0, 50.0, 29.0)
        rh_today = st.number_input("Avg RH Today (%)", 0.0, 100.0, 65.0)
        co_today = st.number_input("Avg CO (ppm)", 0.0, 2000.0, 400.0)
        nh3_today = st.number_input("Avg NH₃ (ppm)", 0.0, 200.0, 22.0)

    submitted = st.form_submit_button("Predict")

# -------------------------------------------------------
# Predictions Section
# -------------------------------------------------------
if submitted:

    # -------------------------------------------------------
    # Auto-Calculated Metrics
    # -------------------------------------------------------
    avg_temp_7d = history["temp"].mean()
    avg_rh_7d = history["rh"].mean()
    avg_co_7d = history["co"].mean()
    avg_nh3_7d = history["nh3"].mean()
    feed_7d = history["feed"].sum()
    sample_weight_7d = history["weight"].mean()

    mort_l1, mort_l2, mort_l3 = history["mortality"].tail(3).tolist()
    feed_l1, feed_l2, feed_l3 = history["feed"].tail(3).tolist()

    # -------------------------------------------------------
    # BEAUTIFUL CARD UI FOR AUTO-CALCULATED DATA
    # -------------------------------------------------------
    st.markdown("""
        <style>
            .card {
                background-color: #ffffff;
                padding: 18px;
                border-radius: 12px;
                box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .card h4 {
                margin: 0;
                color: #007bff;
                padding-bottom: 8px;
            }
            .card p {
                margin: 2px 0;
                font-size: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("## 📌 Auto-Calculated Farm Metrics (Read Only)")

    c1, c2 = st.columns(2)

    # LEFT CARD → Environmental
    c1.markdown(f"""
        <div class="card">
            <h4>🌦 Environmental Averages (7 Days)</h4>
            <p>🌡️ <b>Avg Temp:</b> {avg_temp_7d:.2f} °C</p>
            <p>💧 <b>Avg RH:</b> {avg_rh_7d:.2f} %</p>
            <p>🫁 <b>Avg CO:</b> {avg_co_7d:.0f} ppm</p>
            <p>🟤 <b>Avg NH₃:</b> {avg_nh3_7d:.1f} ppm</p>
        </div>
    """, unsafe_allow_html=True)

    # RIGHT CARD → Feed & Weight
    c2.markdown(f"""
        <div class="card">
            <h4>🍽 Feed & Weight Summary</h4>
            <p>🪵 <b>Total Feed (7d):</b> {feed_7d:.1f} kg</p>
            <p>⚖️ <b>Avg Sample Weight (7d):</b> {sample_weight_7d:.2f} kg</p>
        </div>
    """, unsafe_allow_html=True)

    # LAG CARD FULL WIDTH
    st.markdown(f"""
        <div class="card">
            <h4>⏳ Lag Values (Last 3 Days)</h4>
            <p>☠️ <b>Mortality:</b> {mort_l1}, {mort_l2}, {mort_l3}</p>
            <p>🍽️ <b>Feed (kg):</b> {feed_l1:.1f}, {feed_l2:.1f}, {feed_l3:.1f}</p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------
    # Build Row for Prediction
    # -------------------------------------------------------
    row = pd.DataFrame([{
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality_today,
        "feed_kg": feed_today,
        "water_consumption_l": water_today,
        "avg_temp_c": temp_today,
        "avg_rh": rh_today,
        "avg_co_ppm": co_today,
        "avg_nh_ppm": nh3_today,
        "sample_weight": sample_weight_today,

        "avg_temp_c_7d": avg_temp_7d,
        "avg_rh_7d": avg_rh_7d,
        "avg_co_ppm_7d": avg_co_7d,
        "avg_nh_ppm_7d": avg_nh3_7d,
        "feed_kg_7d": feed_7d,
        "sample_weight_7d": sample_weight_7d,

        "mortality_lag1": mort_l1,
        "mortality_lag2": mort_l2,
        "mortality_lag3": mort_l3,
        "feed_kg_lag1": feed_l1,
        "feed_kg_lag2": feed_l2,
        "feed_kg_lag3": feed_l3,

        "fcr_today": feed_today / sample_weight_today if sample_weight_today > 0 else 1.8
    }])

    # -------------------------------------------------------
    # Predictions
    # -------------------------------------------------------
    pred_weight = weight_model.predict(row)[0]
    pred_mortality = mortality_model.predict(row)[0]
    pred_fcr = fcr_model.predict(row)[0]

    # -------------------------------------------------------
    # Display Predictions
    # -------------------------------------------------------
    st.markdown("## 📊 AI Predictions")

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Weight (kg)", f"{pred_weight:.3f}")
    p2.metric("Predicted Mortality (birds)", f"{round(pred_mortality)}")
    p3.metric("Predicted FCR", f"{pred_fcr:.3f}")
