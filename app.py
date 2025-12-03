import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# Load models
# -----------------------------
weight_model = pickle.load(open("weight_model.pkl", "rb"))
mortality_model = pickle.load(open("mortality_model.pkl", "rb"))
fcr_model = pickle.load(open("fcr_model.pkl", "rb"))

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="iPoultry AI Assistant", layout="wide")
st.title("🐔 iPoultry AI — Daily Farm Predictions")

st.markdown("### Enter today's flock metrics")

# -----------------------------
# Farmer Input Form
# -----------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age_in_days = st.number_input("Age in Days", 0, 100, 14)
        birds_alive = st.number_input("Birds Alive", 0, 100000, 900)
        mortality_today = st.number_input("Mortality Today", 0, 1000, 1)

    with col2:
        feed_today = st.number_input("Feed Today (kg)", 0.0, 500.0, 22.0)
        water_today = st.number_input("Water Today (L)", 0.0, 500.0, 30.0)
        sample_weight_today = st.number_input("Sample Weight (kg)", 0.0, 5.0, 1.2)

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Auto-calculation & Prediction
# -----------------------------
if submitted:

    # Simulate history (replace with real DB in production)
    np.random.seed(42)
    history = pd.DataFrame({
        "temp": np.random.uniform(27, 30, 7),
        "rh": np.random.uniform(55, 70, 7),
        "co": np.random.uniform(300, 500, 7),
        "nh3": np.random.uniform(10, 25, 7),
        "feed": np.random.uniform(18, 25, 7),
        "mortality": np.random.randint(0, 5, 7),
        "weight": np.random.uniform(0.8, 1.5, 7),
    })

    # 7-day averages
    avg_temp_7d = history["temp"].mean()
    avg_rh_7d = history["rh"].mean()
    avg_co_7d = history["co"].mean()
    avg_nh3_7d = history["nh3"].mean()
    feed_7d = history["feed"].sum()
    sample_weight_7d = history["weight"].mean()
    fcr_7d = feed_7d / sample_weight_7d if sample_weight_7d > 0 else 0

    # Lags (last 3 days)
    mort_l1, mort_l2, mort_l3 = history["mortality"].tail(3).tolist()
    feed_l1, feed_l2, feed_l3 = history["feed"].tail(3).tolist()

    # Compute FCR for today
    fcr_today = feed_today / sample_weight_today if sample_weight_today > 0 else 0

    # Build row for prediction
    row = pd.DataFrame([{
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality_today,
        "feed_kg": feed_today,
        "water_consumption_l": water_today,
        "avg_temp_c": avg_temp_7d,  # use today's as placeholder if needed
        "avg_rh": avg_rh_7d,
        "avg_co_ppm": avg_co_7d,
        "avg_nh_ppm": avg_nh3_7d,
        "sample_weight": sample_weight_today,
        "fcr_today": fcr_today,
        "avg_temp_c_7d": avg_temp_7d,
        "avg_rh_7d": avg_rh_7d,
        "avg_co_ppm_7d": avg_co_7d,
        "avg_nh_ppm_7d": avg_nh3_7d,
        "feed_kg_7d": feed_7d,
        "sample_weight_7d": sample_weight_7d,
        "fcr_today_7d": fcr_7d,
        "mortality_lag1": mort_l1,
        "mortality_lag2": mort_l2,
        "mortality_lag3": mort_l3,
        "feed_kg_lag1": feed_l1,
        "feed_kg_lag2": feed_l2,
        "feed_kg_lag3": feed_l3
    }])

    # Predictions
    pred_weight = weight_model.predict(row)[0]
    pred_mortality = mortality_model.predict(row)[0]
    pred_fcr = fcr_model.predict(row)[0]

    # -----------------------------
    # Display Results Beautifully
    # -----------------------------
    st.subheader("📊 AI Predictions for Tomorrow")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Avg Weight (kg)", f"{pred_weight:.3f}")
    c2.metric("Predicted Mortality (birds)", f"{int(round(pred_mortality))}")
    c3.metric("Predicted FCR", f"{pred_fcr:.3f}")

    # Display auto-calculated metrics for reference
    st.markdown("### 📌 Auto-Calculated Metrics")
    r1, r2, r3 = st.columns(3)
    r1.metric("7-Day Avg Temp (°C)", f"{avg_temp_7d:.2f}")
    r2.metric("7-Day Avg RH (%)", f"{avg_rh_7d:.2f}")
    r3.metric("7-Day Avg CO (ppm)", f"{avg_co_7d:.0f}")
    r4, r5, r6 = st.columns(3)
    r4.metric("7-Day Avg NH₃ (ppm)", f"{avg_nh3_7d:.1f}")
    r5.metric("Feed Total 7d (kg)", f"{feed_7d:.1f}")
    r6.metric("Sample Weight 7d (kg)", f"{sample_weight_7d:.2f}")
