import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# -----------------------------
# Load models
# -----------------------------
weight_model = pickle.load(open("weight_model.pkl", "rb"))
mortality_model = pickle.load(open("mortality_model.pkl", "rb"))
fcr_model = pickle.load(open("fcr_model.pkl", "rb"))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="iPoultry AI", layout="wide")
st.title("🐔 iPoultry AI — Daily Farm Predictions")

st.markdown("### Enter today’s flock metrics")

# -----------------------------
# Farmer Inputs
# -----------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age_in_days = st.number_input("Age (Days)", 0, 100, 14)
        birds_alive = st.number_input("Birds Alive", 0, 200000, 900)
        mortality_today = st.number_input("Mortality Today", 0, 1000, 1)
        feed_today = st.number_input("Feed Today (kg)", 0.0, 500.0, 22.0)

    with col2:
        water_today = st.number_input("Water Today (L)", 0.0, 500.0, 30.0)
        sample_weight_today = st.number_input("Sample Weight (kg)", 0.0, 5.0, 1.2)

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Prediction and Display
# -----------------------------
if submitted:
    # -------------------------
    # Simulate last 7 days for demo (replace with DB in real use)
    # -------------------------
    np.random.seed(42)
    history_days = 7
    history = pd.DataFrame({
        "temp": np.random.uniform(27, 30, history_days),
        "rh": np.random.uniform(55, 70, history_days),
        "co": np.random.uniform(300, 500, history_days),
        "nh3": np.random.uniform(10, 25, history_days),
        "feed": np.random.uniform(18, 25, history_days),
        "mortality": np.random.randint(0, 5, history_days),
        "weight": np.random.uniform(0.8, 1.5, history_days),
    })

    # -------------------------
    # Compute 7-day averages
    # -------------------------
    avg_temp_7d = history["temp"].mean()
    avg_rh_7d = history["rh"].mean()
    avg_co_7d = history["co"].mean()
    avg_nh3_7d = history["nh3"].mean()
    feed_7d = history["feed"].sum()
    sample_weight_7d = history["weight"].mean()
    fcr_7d = feed_7d / sample_weight_7d if sample_weight_7d > 0 else 0

    # -------------------------
    # Compute last 3-day lags
    # -------------------------
    mort_lags = history["mortality"].tail(3).tolist()
    feed_lags = history["feed"].tail(3).tolist()
    # Pad if less than 3 days
    while len(mort_lags) < 3: mort_lags.insert(0,0)
    while len(feed_lags) < 3: feed_lags.insert(0,0)

    # -------------------------
    # Compute FCR today
    # -------------------------
    fcr_today = feed_today / sample_weight_today if sample_weight_today > 0 else 0

    # -------------------------
    # Build DataFrame for prediction
    # -------------------------
    row = pd.DataFrame([{
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality_today,
        "feed_kg": feed_today,
        "water_consumption_l": water_today,
        "avg_temp_c": np.mean([avg_temp_7d, avg_temp_7d]),  # placeholder
        "avg_rh": np.mean([avg_rh_7d, avg_rh_7d]),
        "avg_co_ppm": np.mean([avg_co_7d, avg_co_7d]),
        "avg_nh_ppm": np.mean([avg_nh3_7d, avg_nh3_7d]),
        "sample_weight": sample_weight_today,
        "fcr_today": fcr_today,
        "avg_temp_c_7d": avg_temp_7d,
        "avg_rh_7d": avg_rh_7d,
        "avg_co_ppm_7d": avg_co_7d,
        "avg_nh_ppm_7d": avg_nh3_7d,
        "feed_kg_7d": feed_7d,
        "sample_weight_7d": sample_weight_7d,
        "fcr_today_7d": fcr_7d,
        "mortality_lag1": mort_lags[-1],
        "mortality_lag2": mort_lags[-2],
        "mortality_lag3": mort_lags[-3],
        "feed_kg_lag1": feed_lags[-1],
        "feed_kg_lag2": feed_lags[-2],
        "feed_kg_lag3": feed_lags[-3]
    }])

    # -------------------------
    # Reorder columns to match model
    # -------------------------
    row_w = row[weight_model.feature_names_in_]
    row_m = row[mortality_model.feature_names_in_]
    row_f = row[fcr_model.feature_names_in_]

    # -------------------------
    # Make predictions
    # -------------------------
    pred_weight = weight_model.predict(row_w)[0]
    pred_mortality = mortality_model.predict(row_m)[0]
    pred_fcr = fcr_model.predict(row_f)[0]
    #pred_fcr = min(pred_fcr, 3.0)  # Cap FCR to 3.0 for realism

    # -------------------------
    # Display nicely
    # -------------------------
    st.subheader("📊 AI Predictions for Tomorrow")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Weight (kg)", f"{pred_weight:.3f}")
    col2.metric("Predicted Mortality (birds)", f"{int(round(pred_mortality))}")
    col3.metric("Predicted FCR", f"{pred_fcr:.3f}")

    st.markdown("### 📌 Auto-calculated Metrics (from last 7 days)")
    st.write(f"**7-day Avg Temp:** {avg_temp_7d:.2f} °C  |  **7-day Avg RH:** {avg_rh_7d:.2f} %")
    st.write(f"**7-day Avg CO:** {avg_co_7d:.0f} ppm  |  **7-day Avg NH₃:** {avg_nh3_7d:.1f} ppm")
    st.write(f"**7-day Total Feed:** {feed_7d:.1f} kg  |  **7-day Avg Sample Weight:** {sample_weight_7d:.2f} kg")
    st.write(f"**7-day FCR:** {fcr_7d:.3f}")
    st.write(f"**Mortality Lags (last 3 days):** {mort_lags[-3]}, {mort_lags[-2]}, {mort_lags[-1]}")
    st.write(f"**Feed Lags (last 3 days):** {feed_lags[-3]:.1f}, {feed_lags[-2]:.1f}, {feed_lags[-1]:.1f}")
