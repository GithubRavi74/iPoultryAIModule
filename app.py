import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import traceback

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
# Farmer Inputs (only these 6)
# -----------------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age_in_days = st.number_input("Age (Days)", 0, 100, 14, key="age_in_days")
        birds_alive = st.number_input("Birds Alive", 0, 200000, 900, key="birds_alive")
        mortality_today = st.number_input("Mortality Today", 0, 1000, 1, key="mortality_today")
        feed_today = st.number_input("Feed Today (kg)", 0.0, 500.0, 22.0, key="feed_today")

    with col2:
        water_today = st.number_input("Water Today (L)", 0.0, 500.0, 30.0, key="water_today")
        sample_weight_today = st.number_input("Sample Weight (kg)", 0.0, 5.0, 1.2, key="sample_weight_today")

    submitted = st.form_submit_button("Predict")

# -----------------------------
# When user clicks Predict
# -----------------------------
if submitted:
    try:
        # -------------------------
        # Simulate last 7 days for demo (replace with DB in production)
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
        while len(mort_lags) < 3: mort_lags.insert(0, 0)
        while len(feed_lags) < 3: feed_lags.insert(0, 0)

        # -------------------------
        # Compute FCR today
        # -------------------------
        fcr_today = feed_today / sample_weight_today if sample_weight_today > 0 else 0

        # -------------------------
        # Forecast settings
        # -------------------------
        forecast_days = 33
        future_ages = np.arange(age_in_days, age_in_days + forecast_days)

        # Lists to hold predictions
        weight_preds = []
        mort_preds = []
        fcr_preds = []

        # -------------------------
        # Iterative (day-by-day) forecasting until target ages
        # We'll use static placeholders for environmental features (replace with sensor/history)
        # -------------------------
        for future_age in future_ages:
            # Build a single-row DataFrame for the model
            row = pd.DataFrame([{
                # Use future_age here so prediction corresponds to that age
                "age_in_days": int(future_age),
                "birds_alive": birds_alive,
                "mortality": mortality_today,
                "feed_kg": feed_today,
                "water_consumption_l": water_today,
                # placeholders for environmental features (use real sensors/history if available)
                "avg_temp_c": float(avg_temp_7d),
                "avg_rh": float(avg_rh_7d),
                "avg_co_ppm": float(avg_co_7d),
                "avg_nh_ppm": float(avg_nh3_7d),
                "sample_weight": float(sample_weight_today),
                "fcr_today": float(fcr_today),
                # 7-day aggregated placeholders
                "avg_temp_c_7d": float(avg_temp_7d),
                "avg_rh_7d": float(avg_rh_7d),
                "avg_co_ppm_7d": float(avg_co_7d),
                "avg_nh_ppm_7d": float(avg_nh3_7d),
                "feed_kg_7d": float(feed_7d),
                "sample_weight_7d": float(sample_weight_7d),
                "fcr_today_7d": float(fcr_7d),
                # lag placeholders
                "mortality_lag1": int(mort_lags[-1]),
                "mortality_lag2": int(mort_lags[-2]),
                "mortality_lag3": int(mort_lags[-3]),
                "feed_kg_lag1": float(feed_lags[-1]),
                "feed_kg_lag2": float(feed_lags[-2]),
                "feed_kg_lag3": float(feed_lags[-3])
            }])

            # Reorder/select columns exactly as each model expects
            row_w = row[weight_model.feature_names_in_]
            row_m = row[mortality_model.feature_names_in_]
            row_f = row[fcr_model.feature_names_in_]

            # Predict
            w = weight_model.predict(row_w)[0]
            m = mortality_model.predict(row_m)[0]
            f = fcr_model.predict(row_f)[0]

            # Append
            weight_preds.append(float(w))
            mort_preds.append(float(m))
            fcr_preds.append(float(f))

            # Optional: update lag placeholders for next iteration in a simple way
            # (we keep them constant here; for more realistic simulation you can update them using predictions)
            # e.g. mortality_today = int(round(m)) ; sample_weight_today = w  (if you want)
            # We'll not overwrite original inputs to preserve farmer-provided values.

        # -------------------------
        # Build dataframe for display
        # -------------------------
        df_forecast = pd.DataFrame({
            "Bird Age (days)": future_ages.astype(int),
            "Predicted Weight (kg)": np.round(weight_preds, 3),
            "Predicted Mortality (birds)": np.round(mort_preds).astype(int),
            "Predicted FCR": np.round(fcr_preds, 3)
        })

        # -------------------------
        # Display results
        # -------------------------
        st.subheader(f"📈 {forecast_days}-Day Forecast (from age {age_in_days} to {age_in_days + forecast_days - 1})")
        st.dataframe(df_forecast, use_container_width=True)

        st.markdown("### Weight Forecast")
        st.line_chart(df_forecast.set_index("Bird Age (days)")["Predicted Weight (kg)"])

        st.markdown("### Summary (Auto-calculated from last 7 days)")
        st.write(f"7-day Avg Temp: {avg_temp_7d:.2f} °C   |   7-day Avg RH: {avg_rh_7d:.2f} %")
        st.write(f"7-day Avg CO: {avg_co_7d:.0f} ppm   |   7-day Avg NH₃: {avg_nh3_7d:.1f} ppm")
        st.write(f"7-day Total Feed: {feed_7d:.1f} kg   |   7-day Avg Sample Weight: {sample_weight_7d:.2f} kg")
        st.write(f"7-day FCR: {fcr_7d:.3f}")
        st.write(f"Mortality Lags (last 3 days): {mort_lags[-3]}, {mort_lags[-2]}, {mort_lags[-1]}")
        st.write(f"Feed Lags (last 3 days): {feed_lags[-3]:.1f}, {feed_lags[-2]:.1f}, {feed_lags[-1]:.1f}")

    except Exception as e:
        st.error("Prediction failed — see logs for details.")
        st.error(str(e))
        # print full traceback to the server logs (not exposed to user)
        traceback.print_exc()
