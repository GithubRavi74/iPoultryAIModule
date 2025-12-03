import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Load Models
# -----------------------------
weight_model = pickle.load(open("weight_model.pkl", "rb"))
mortality_model = pickle.load(open("mortality_model.pkl", "rb"))
fcr_model = pickle.load(open("fcr_model.pkl", "rb"))

# -----------------------------
# Load or Create History File
# -----------------------------
HISTORY_FILE = "history.csv"

if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
else:
    history = pd.DataFrame(columns=[
        "age_in_days", "birds_alive", "mortality", "feed_kg", "water_l",
        "temp", "rh", "co", "nh3", "sample_weight"
    ])
    history.to_csv(HISTORY_FILE, index=False)

# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(page_title="iPoultry AI", layout="wide")
st.title("🐔 iPoultry AI — Daily Farm Predictions")

st.markdown("### Enter today’s flock metrics")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age_in_days = st.number_input("Age (Days)", 0, 100, 14)
        birds_alive = st.number_input("Birds Alive", 0, 200000, 9000)
        mortality_today = st.number_input("Mortality Today", 0, 10000, 5)
        feed_kg = st.number_input("Feed Given Today (kg)", 0.0, 2000.0, 200.0)
        water_l = st.number_input("Water Consumed Today (L)", 0.0, 5000.0, 150.0)

    with col2:
        temp = st.number_input("Average Temp (°C)", 0.0, 50.0, 29.0)
        rh = st.number_input("Average RH (%)", 0.0, 100.0, 65.0)
        co = st.number_input("CO (ppm)", 0.0, 2000.0, 350.0)
        nh3 = st.number_input("NH₃ (ppm)", 0.0, 200.0, 18.0)
        sample_weight = st.number_input("Sample Weight (kg)", 0.0, 5.0, 1.2)

    submitted = st.form_submit_button("Predict")

# -----------------------------
# Prediction Logic
# -----------------------------
if submitted:

    # -------------------------
    # Store today’s data in history
    # -------------------------
    today_row = {
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality_today,
        "feed_kg": feed_kg,
        "water_l": water_l,
        "temp": temp,
        "rh": rh,
        "co": co,
        "nh3": nh3,
        "sample_weight": sample_weight
    }

    history = history.append(today_row, ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)

    # -------------------------
    # Compute Lags (last 1, 2, 3 days)
    # -------------------------
    def get_lag(col):
        l1 = history[col].shift(1).iloc[-1] if len(history)>=2 else 0
        l2 = history[col].shift(2).iloc[-1] if len(history)>=3 else 0
        l3 = history[col].shift(3).iloc[-1] if len(history)>=4 else 0
        return l1, l2, l3

    mort_l1, mort_l2, mort_l3 = get_lag("mortality")
    feed_l1, feed_l2, feed_l3 = get_lag("feed_kg")

    # -------------------------
    # Compute 7-Day Averages (auto)
    # -------------------------
    last7 = history.tail(7)

    avg_temp_7d = last7["temp"].mean()
    avg_rh_7d = last7["rh"].mean()
    avg_co_7d = last7["co"].mean()
    avg_nh3_7d = last7["nh3"].mean()
    feed_7d = last7["feed_kg"].sum()
    sample_wt_7d = last7["sample_weight"].mean()

    # -------------------------
    # Compute FCR Automatically
    # -------------------------
    # Avoid division by zero
    if sample_weight > 0:
        fcr_today = feed_kg / sample_weight
    else:
        fcr_today = 0

    # -------------------------
    # Build Final Row for Model
    # -------------------------
    row = pd.DataFrame([{
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality_today,
        "feed_kg": feed_kg,
        "water_consumption_l": water_l,
        "avg_temp_c": temp,
        "avg_rh": rh,
        "avg_co_ppm": co,
        "avg_nh_ppm": nh3,
        "sample_weight": sample_weight,
        "fcr_today": fcr_today,
        "avg_temp_c_7d": avg_temp_7d,
        "avg_rh_7d": avg_rh_7d,
        "avg_co_ppm_7d": avg_co_7d,
        "avg_nh_ppm_7d": avg_nh3_7d,
        "feed_kg_7d": feed_7d,
        "sample_weight_7d": sample_wt_7d,
        "fcr_today_7d": feed_7d / sample_wt_7d if sample_wt_7d > 0 else 0,
        "mortality_lag1": mort_l1,
        "mortality_lag2": mort_l2,
        "mortality_lag3": mort_l3,
        "feed_kg_lag1": feed_l1,
        "feed_kg_lag2": feed_l2,
        "feed_kg_lag3": feed_l3
    }])

    # -------------------------
    # Predict
    # -------------------------
    pred_weight = weight_model.predict(row)[0]
    pred_mortality = mortality_model.predict(row)[0]
    pred_fcr = fcr_model.predict(row)[0]

    # -------------------------
    # Display
    # -------------------------
    st.subheader("📊 AI Predictions for Tomorrow")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Avg Weight (kg)", f"{pred_weight:.3f}")
    c2.metric("Predicted Mortality (birds)", f"{int(round(pred_mortality))}")
    c3.metric("Predicted FCR", f"{pred_fcr:.3f}")
