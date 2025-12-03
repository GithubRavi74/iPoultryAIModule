import streamlit as st
import pandas as pd
import pickle
import os

# ---------------------------------------------
# Load Models
# ---------------------------------------------
weight_model = pickle.load(open("weight_model.pkl", "rb"))
mortality_model = pickle.load(open("mortality_model.pkl", "rb"))
fcr_model = pickle.load(open("fcr_model.pkl", "rb"))

# ---------------------------------------------
# History
# ---------------------------------------------
HISTORY_FILE = "history.csv"

if os.path.exists(HISTORY_FILE):
    history = pd.read_csv(HISTORY_FILE)
else:
    history = pd.DataFrame(columns=[
        "age_in_days", "birds_alive", "mortality", "feed_kg",
        "water_l", "temp", "rh", "co", "nh3", "sample_weight"
    ])
    history.to_csv(HISTORY_FILE, index=False)

# ---------------------------------------------
# UI CONFIG
# ---------------------------------------------
st.set_page_config(page_title="iPoultry AI Module", layout="wide")
st.title("🐔 iPoultry AI — Daily Predictions Dashboard")

st.markdown("""
Enter today's flock metrics.  
All advanced calculations like 7-day averages, lags, and FCR will be computed automatically.
""")

# ---------------------------------------------
# INPUT SECTION
# ---------------------------------------------
with st.form("farm_inputs"):
    col1, col2 = st.columns(2)

    with col1:
        age_in_days = st.number_input("Age in Days", 0, 100, 14)
        birds_alive = st.number_input("Birds Alive", 0, 200000, 9000)
        mortality_today = st.number_input("Mortality Today", 0, 10000, 5)

    with col2:
        feed_kg = st.number_input("Feed Given Today (kg)", 0.0, 2000.0, 200.0)
        water_l = st.number_input("Water Consumed Today (L)", 0.0, 5000.0, 150.0)
        sample_weight = st.number_input("Sample Weight (kg)", 0.0, 5.0, 1.2)

    submitted = st.form_submit_button("Predict")

# ---------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------
if submitted:

    # Save today's input
    today = {
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality_today,
        "feed_kg": feed_kg,
        "water_l": water_l,
        "temp": 0,    # placeholder because environment removed
        "rh": 0,
        "co": 0,
        "nh3": 0,
        "sample_weight": sample_weight
    }

    history = history.append(today, ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)

    # -----------------------------------------
    # Lags (auto)
    # -----------------------------------------
    def lag(col):
        if len(history) < 2:
            return 0, 0, 0
        l1 = history[col].shift(1).iloc[-1]
        l2 = history[col].shift(2).iloc[-1] if len(history) >= 3 else 0
        l3 = history[col].shift(3).iloc[-1] if len(history) >= 4 else 0
        return l1, l2, l3

    mort_l1, mort_l2, mort_l3 = lag("mortality")
    feed_l1, feed_l2, feed_l3 = lag("feed_kg")

    # -----------------------------------------
    # 7-day averages (Auto)
    # -----------------------------------------
    last7 = history.tail(7)

    sample_weight_7d = last7["sample_weight"].mean()
    feed_7d = last7["feed_kg"].sum()

    # FCR auto
    fcr_today = feed_kg / sample_weight if sample_weight > 0 else 0
    fcr_7d = feed_7d / sample_weight_7d if sample_weight_7d > 0 else 0

    # -----------------------------------------
    # Construct final prediction row matching your model
    # -----------------------------------------
    row = pd.DataFrame([{
        "age_in_days": age_in_days,
        "birds_alive": birds_alive,
        "mortality": mortality_today,
        "feed_kg": feed_kg,
        "water_consumption_l": water_l,

        "avg_temp_c": 0,
        "avg_rh": 0,
        "avg_co_ppm": 0,
        "avg_nh_ppm": 0,

        "sample_weight": sample_weight,
        "fcr_today": fcr_today,

        "avg_temp_c_7d": 0,
        "avg_rh_7d": 0,
        "avg_co_ppm_7d": 0,
        "avg_nh_ppm_7d": 0,

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

    # -----------------------------------------
    # Predictions
    # -----------------------------------------
    pred_weight = float(weight_model.predict(row)[0])
    pred_mortality = float(mortality_model.predict(row)[0])
    pred_fcr = float(fcr_model.predict(row)[0])

    # -----------------------------------------
    # SHOW PREDICTIONS FIRST
    # -----------------------------------------
    st.subheader("📊 AI Predictions for Tomorrow")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Avg Weight (kg)", f"{pred_weight:.3f}")
    c2.metric("Predicted Mortality (birds)", f"{round(pred_mortality)}")
    c3.metric("Predicted FCR", f"{pred_fcr:.3f}")

    st.markdown("---")

    # -----------------------------------------
    # BELOW: BEAUTIFUL FARM ANALYTICS CARDS
    # -----------------------------------------
    st.subheader("📘 Auto-calculated Farm Insights")

    # LAG CARDS
    st.markdown("### ➤ Mortality Trend (Lag)")
    l1, l2, l3 = st.columns(3)
    l1.info(f"Yesterday: **{mort_l1} birds**")
    l2.info(f"2 Days Ago: **{mort_l2} birds**")
    l3.info(f"3 Days Ago: **{mort_l3} birds**")

    st.markdown("### ➤ Feed Trend (Lag)")
    f1, f2, f3 = st.columns(3)
    f1.success(f"Yesterday Feed: **{feed_l1} kg**")
    f2.success(f"2 Days Ago: **{feed_l2} kg**")
    f3.success(f"3 Days Ago: **{feed_l3} kg**")

    st.markdown("### ➤ 7-Day Performance Summary")
    a1, a2, a3 = st.columns(3)
    a1.metric("7-Day Avg Sample Weight (kg)", f"{sample_weight_7d:.3f}")
    a2.metric("7-Day Total Feed (kg)", f"{feed_7d:.1f}")
    a3.metric("7-Day FCR", f"{fcr_7d:.3f}")
