import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load models
weight_model = pickle.load(open("weight_model.pkl", "rb"))
mortality_model = pickle.load(open("mortality_model.pkl", "rb"))
fcr_model = pickle.load(open("fcr_model.pkl", "rb"))

# Page settings
st.set_page_config(page_title="iPoultry AI Assistant", layout="wide")
st.title("🐔 iPoultry AI — Weight, Mortality & FCR Prediction")

st.markdown("### Enter today's flock metrics")

# --------------------------------------------------------
# Helper functions to simulate historical data (for demo)
# --------------------------------------------------------
def generate_demo_history(days=7):
    """Simulated last 7 days of farm data so UI calculations work.
       Replace with real database values later.
    """
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

history = generate_demo_history(7)   # replace with DB data

# --------------------------------------------------------
# Farmer Input Section (TODAY ONLY)
# --------------------------------------------------------
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

# --------------------------------------------------------
# When Predict is clicked
# --------------------------------------------------------
if submitted:

    # -----------------------------------------------
    # Auto-calculate 7-day averages from history
    # -----------------------------------------------
    avg_temp_7d = history["temp"].mean()
    avg_rh_7d = history["rh"].mean()
    avg_co_7d = history["co"].mean()
    avg_nh3_7d = history["nh3"].mean()
    feed_7d = history["feed"].sum()
    sample_weight_7d = history["weight"].mean()

    # -----------------------------------------------
    # Auto-generate lag features (last 3 days)
    # -----------------------------------------------
    mort_l1, mort_l2, mort_l3 = history["mortality"].tail(3).tolist()
    feed_l1, feed_l2, feed_l3 = history["feed"].tail(3).tolist()

    # -----------------------------------------------
    # Show the auto-calculated values (read-only)
    # -----------------------------------------------
    st.markdown("## 📌 Auto-Calculated Farm Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Temp (7d)", f"{avg_temp_7d:.2f} °C")
    c2.metric("Avg RH (7d)", f"{avg_rh_7d:.2f} %")
    c3.metric("Avg CO (7d)", f"{avg_co_7d:.0f} ppm")
    c4.metric("Avg NH₃ (7d)", f"{avg_nh3_7d:.1f} ppm")
    c5.metric("Feed Total (7d)", f"{feed_7d:.1f} kg")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Sample Weight (7d)", f"{sample_weight_7d:.2f} kg")
    c7.metric("Mortality Lag-1", f"{mort_l1}")
    c8.metric("Mortality Lag-2", f"{mort_l2}")
    c9.metric("Mortality Lag-3", f"{mort_l3}")
    c10.metric("Feed Lag-1 / -2 / -3", f"{feed_l1:.1f} / {feed_l2:.1f} / {feed_l3:.1f}")

    # -----------------------------------------------
    # Build final row for model
    # -----------------------------------------------
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

        # Auto 7-day values
        "avg_temp_c_7d": avg_temp_7d,
        "avg_rh_7d": avg_rh_7d,
        "avg_co_ppm_7d": avg_co_7d,
        "avg_nh_ppm_7d": avg_nh3_7d,
        "feed_kg_7d": feed_7d,
        "sample_weight_7d": sample_weight_7d,

        # Lag values
        "mortality_lag1": mort_l1,
        "mortality_lag2": mort_l2,
        "mortality_lag3": mort_l3,
        "feed_kg_lag1": feed_l1,
        "feed_kg_lag2": feed_l2,
        "feed_kg_lag3": feed_l3,

        # Computed FCR (optional)
        "fcr_today": feed_today / sample_weight_today if sample_weight_today > 0 else 1.8
    }])

    # -----------------------------------------------
    # Run model predictions
    # -----------------------------------------------
    pred_weight = weight_model.predict(row)[0]
    pred_mortality = mortality_model.predict(row)[0]
    pred_fcr = fcr_model.predict(row)[0]

    # -----------------------------------------------
    # Display clean predictions
    # -----------------------------------------------
    st.markdown("## 📊 AI Predictions")

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted Weight (kg)", f"{pred_weight:.3f}")
    p2.metric("Predicted Mortality (birds)", f"{round(pred_mortality)}")   # rounded
    p3.metric("Predicted FCR", f"{pred_fcr:.3f}")
