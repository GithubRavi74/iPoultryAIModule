import streamlit as st
import pandas as pd
import pickle

# Load models
weight_model = pickle.load(open("weight_model.pkl", "rb"))
mortality_model = pickle.load(open("mortality_model.pkl", "rb"))
fcr_model = pickle.load(open("fcr_model.pkl", "rb"))

st.set_page_config(page_title="iPoultry AI Assistant", layout="wide")
st.title("🐔 iPoultry AI — Weight, Mortality & FCR Prediction")

st.markdown("### Enter today's flock metrics to get AI-powered predictions")

# --- Input Form ---
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age_in_days = st.number_input("Age in Days", 0, 100, 14)
        birds_alive = st.number_input("Birds Alive", 0, 100000, 900)
        mortality = st.number_input("Mortality Today", 0, 1000, 1)
        feed_kg = st.number_input("Feed (kg)", 0.0, 500.0, 22.0)
        water_consumption_l = st.number_input("Water (L)", 0.0, 500.0, 30.0)
        avg_temp_c = st.number_input("Avg Temp (°C)", 0.0, 50.0, 29.0)

    with col2:
        avg_rh = st.number_input("Avg RH (%)", 0.0, 100.0, 65.0)
        avg_co_ppm = st.number_input("Avg CO (ppm)", 0.0, 2000.0, 400.0)
        avg_nh_ppm = st.number_input("Avg NH₃ (ppm)", 0.0, 200.0, 22.0)
        sample_weight = st.number_input("Sample Weight (kg)", 0.0, 5.0, 1.2)
        fcr_today = st.number_input("FCR Today", 0.0, 5.0, 1.7)

    with col3:
        avg_temp_c_7d = st.number_input("Avg Temp (7d)", 0.0, 50.0, 28.0)
        avg_rh_7d = st.number_input("Avg RH (7d)", 0.0, 100.0, 60.0)
        avg_co_ppm_7d = st.number_input("Avg CO (7d)", 0.0, 2000.0, 350.0)
        avg_nh_ppm_7d = st.number_input("Avg NH₃ (7d)", 0.0, 200.0, 20.0)
        feed_kg_7d = st.number_input("Feed (kg, 7d)", 0.0, 1000.0, 150.0)
        sample_weight_7d = st.number_input("Sample Weight (7d)", 0.0, 5.0, 1.1)
        fcr_today_7d = st.number_input("FCR (7d)", 0.0, 5.0, 1.6)
    
    colA, colB, colC = st.columns(3)
    with colA:
        mortality_lag1 = st.number_input("Mortality lag 1", 0, 100, 0)
        mortality_lag2 = st.number_input("Mortality lag 2", 0, 100, 1)
        mortality_lag3 = st.number_input("Mortality lag 3", 0, 100, 0)
    with colB:
        feed_kg_lag1 = st.number_input("Feed kg lag 1", 0.0, 500.0, 21.0)
        feed_kg_lag2 = st.number_input("Feed kg lag 2", 0.0, 500.0, 20.0)
        feed_kg_lag3 = st.number_input("Feed kg lag 3", 0.0, 500.0, 22.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality,
        "feed_kg": feed_kg,
        "water_consumption_l": water_consumption_l,
        "avg_temp_c": avg_temp_c,
        "avg_rh": avg_rh,
        "avg_co_ppm": avg_co_ppm,
        "avg_nh_ppm": avg_nh_ppm,
        "sample_weight": sample_weight,
        "fcr_today": fcr_today,
        "avg_temp_c_7d": avg_temp_c_7d,
        "avg_rh_7d": avg_rh_7d,
        "avg_co_ppm_7d": avg_co_ppm_7d,
        "avg_nh_ppm_7d": avg_nh_ppm_7d,
        "feed_kg_7d": feed_kg_7d,
        "sample_weight_7d": sample_weight_7d,
        "fcr_today_7d": fcr_today_7d,
        "mortality_lag1": mortality_lag1,
        "feed_kg_lag1": feed_kg_lag1,
        "mortality_lag2": mortality_lag2,
        "feed_kg_lag2": feed_kg_lag2,
        "mortality_lag3": mortality_lag3,
        "feed_kg_lag3": feed_kg_lag3
    }])

    pred_weight = weight_model.predict(row)[0]
    pred_mortality = mortality_model.predict(row)[0]
    pred_fcr = fcr_model.predict(row)[0]

    st.subheader("📊 Predictions")
    col1, col2, col3 = st.columns(3)

    col1.metric("Weight (kg)", f"{pred_weight:.3f}")
    col2.metric("Mortality (birds)", f"{pred_mortality:.3f}")
    col3.metric("FCR", f"{pred_fcr:.3f}")
