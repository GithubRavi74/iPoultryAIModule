import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import traceback
import plotly.graph_objects as go

# -----------------------------
# Load models
# -----------------------------
weight_model = pickle.load(open("weight_model.pkl", "rb"))
mortality_model = pickle.load(open("mortality_model.pkl", "rb"))
fcr_model = pickle.load(open("fcr_model.pkl", "rb"))

# -----------------------------
# Ideal Weight Chart (Ross Broiler)
# -----------------------------
ideal_weight_chart = {
    0: 0.043, 1: 0.061, 2: 0.079, 3: 0.099, 4: 0.122,
    5: 0.148, 6: 0.176, 7: 0.208, 8: 0.242, 9: 0.280,
    10: 0.321, 11: 0.366, 12: 0.414, 13: 0.465, 14: 0.519,
    15: 0.576, 16: 0.637, 17: 0.701, 18: 0.768, 19: 0.837,
    20: 0.910, 21: 0.985, 22: 1.062, 23: 1.142, 24: 1.225,
    25: 1.309, 26: 1.395, 27: 1.483, 28: 1.573, 29: 1.664,
    30: 1.757, 31: 1.851, 32: 1.946, 33: 2.041, 34: 2.138,
    35: 2.235, 36: 2.332, 37: 2.430, 38: 2.527, 39: 2.625,
    40: 2.723
}

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
        age_in_days = st.number_input("Age (Days)", 0, 60, 14, key="age_in_days")
        birds_alive = st.number_input("Birds Alive", 0, 200000, 900, key="birds_alive")
        mortality_today = st.number_input("Mortality Today", 0, 1000, 1, key="mortality_today")
        feed_today = st.number_input("Feed Today (kg)", 0.0, 10000.0, 22.0, key="feed_today")

    with col2:
        water_today = 30.0   # Hardcoded as requested
        sample_weight_today = st.number_input("Bird Weight (kg)", 0.0, 5.0, 1.2, key="sample_weight_today")

    submitted = st.form_submit_button("Predict")

# -----------------------------
# When user clicks Predict
# -----------------------------
if submitted:
    try:
        # -------------------------
        # Simulated 7-day history
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
        # 7-day averages
        # -------------------------
        avg_temp_7d = history["temp"].mean()
        avg_rh_7d = history["rh"].mean()
        avg_co_7d = history["co"].mean()
        avg_nh3_7d = history["nh3"].mean()
        feed_7d = history["feed"].sum()
        sample_weight_7d = history["weight"].mean()
        fcr_7d = feed_7d / sample_weight_7d if sample_weight_7d > 0 else 0

        # -------------------------
        # last 3-day lags
        # -------------------------
        mort_lags = history["mortality"].tail(3).tolist()
        feed_lags = history["feed"].tail(3).tolist()
        while len(mort_lags) < 3: mort_lags.insert(0, 0)
        while len(feed_lags) < 3: feed_lags.insert(0, 0)

        # -------------------------
        # FCR today
        # -------------------------
        fcr_today = feed_today / sample_weight_today if sample_weight_today > 0 else 0

        # -------------------------
        # Forecast days but clipped to ideal max age (40)
        # -------------------------
        forecast_days = 33
        max_ideal_age = max(ideal_weight_chart.keys())
        end_age = min(age_in_days + forecast_days - 1, max_ideal_age)
        future_ages = np.arange(age_in_days, end_age + 1)

        weight_preds = []
        mort_preds = []
        fcr_preds = []

        # -------------------------
        # Predict day-by-day
        # -------------------------
        for future_age in future_ages:

            row = pd.DataFrame([{
                "age_in_days": int(future_age),
                "birds_alive": birds_alive,
                "mortality": mortality_today,
                "feed_kg": feed_today,
                "water_consumption_l": water_today,
                "avg_temp_c": float(avg_temp_7d),
                "avg_rh": float(avg_rh_7d),
                "avg_co_ppm": float(avg_co_7d),
                "avg_nh_ppm": float(avg_nh3_7d),
                "sample_weight": float(sample_weight_today),
                "fcr_today": float(fcr_today),
                "avg_temp_c_7d": float(avg_temp_7d),
                "avg_rh_7d": float(avg_rh_7d),
                "avg_co_ppm_7d": float(avg_co_7d),
                "avg_nh_ppm_7d": float(avg_nh3_7d),
                "feed_kg_7d": float(feed_7d),
                "sample_weight_7d": float(sample_weight_7d),
                "fcr_today_7d": float(fcr_7d),
                "mortality_lag1": int(mort_lags[-1]),
                "mortality_lag2": int(mort_lags[-2]),
                "mortality_lag3": int(mort_lags[-3]),
                "feed_kg_lag1": float(feed_lags[-1]),
                "feed_kg_lag2": float(feed_lags[-2]),
                "feed_kg_lag3": float(feed_lags[-3])
            }])

            # Split model input features
            row_w = row[weight_model.feature_names_in_]
            row_m = row[mortality_model.feature_names_in_]
            row_f = row[fcr_model.feature_names_in_]

            # -------------------------
            # Mortality & FCR predictions
            # -------------------------
            m = mortality_model.predict(row_m)[0]
            mort_preds.append(float(m))

            f = fcr_model.predict(row_f)[0]
            f = max(1.2, min(f, 2.5))
            fcr_preds.append(float(f))

            # -------------------------
            # INTELLIGENT WEIGHT CORRECTION (FULL PATCH)
            # -------------------------
            raw_w = weight_model.predict(row_w)[0]

            ideal_w = ideal_weight_chart.get(int(future_age), None)
            ideal_today = ideal_weight_chart.get(int(age_in_days), None)

            if ideal_today and ideal_today > 0:
                today_scale = sample_weight_today / ideal_today
            else:
                today_scale = 1.0

            scaled_w = raw_w * today_scale

            # Previous predicted or today's actual
            prev_w = weight_preds[-1] if weight_preds else sample_weight_today

            if ideal_w:
                ideal_prev = ideal_weight_chart.get(int(future_age) - 1, prev_w)
                min_growth = max(0.01, ideal_w - ideal_prev)
            else:
                min_growth = 0.02

            scaled_w = max(scaled_w, prev_w + min_growth)

            if ideal_w:
                alpha = 0.65
                final_w = (alpha * ideal_w) + ((1 - alpha) * scaled_w)
            else:
                final_w = scaled_w

            if len(weight_preds) > 0:
                final_w = (0.7 * final_w) + (0.3 * weight_preds[-1])

            weight_preds.append(float(final_w))

        # -------------------------
        # Build dataframe
        # -------------------------
        ideal_list = [ideal_weight_chart.get(int(a)) for a in future_ages]

        df_forecast = pd.DataFrame({
            "Bird Age (days)": future_ages.astype(int),
            "Predicted Weight (kg)": np.round(weight_preds, 3),
            "Ideal Weight (kg)": ideal_list,
            "Difference (kg)": np.round(np.array(weight_preds) - np.array(ideal_list), 3),
            "Predicted Mortality (birds)": np.round(mort_preds).astype(int),
            "Predicted FCR": np.round(fcr_preds, 3)
        })

        # -----------------------------
        # Highlight Key Predictions
        # -----------------------------
        # Get predicted values for Day 33 and Day 35
        pred_33 = df_forecast[df_forecast["Bird Age (days)"] == 33]
        pred_35 = df_forecast[df_forecast["Bird Age (days)"] == 35]

        if not pred_33.empty:
          w33 = pred_33["Predicted Weight (kg)"].values[0]
          #m33 = pred_33["Predicted Mortality (birds)"].values[0]
          #f33 = pred_33["Predicted FCR"].values[0]

        st.markdown(
        f"""
        <h2 style="color:#2E86C1;">Day 33 Prediction:&nbsp;&nbsp;</h2>
        <h3>Weight: {w33:.2f} kg</h3>
        """,
        unsafe_allow_html=True
        )

        if not pred_35.empty:
          w35 = pred_35["Predicted Weight (kg)"].values[0]
          #m35 = pred_35["Predicted Mortality (birds)"].values[0]
          #f35 = pred_35["Predicted FCR"].values[0]

        st.markdown(
        f"""
        <h2 style="color:#27AE60;">Day 35 Prediction:</h2>
        <h3>Weight: <b>{w35:.2f} kg</b> &nbsp;&nbsp;</h3>
        """,
        unsafe_allow_html=True
       )




        # -------------------------
        # Display main section
        # -------------------------
        st.subheader("📈 Bird Weight Prediction")
        st.markdown("#### 🐔 Broiler - Ross")

        st.dataframe(
            df_forecast.style.format({
                "Predicted Weight (kg)": "{:.3f}",
                "Ideal Weight (kg)": "{:.3f}",
                "Difference (kg)": "{:.3f}",
                "Predicted FCR": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # -------------------------
        # Combined Weight Curve
        # -------------------------
        st.subheader("📉 Predicted vs Ideal Weight Curve")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_forecast["Bird Age (days)"],
            y=df_forecast["Predicted Weight (kg)"],
            mode="lines+markers",
            name="Predicted Weight",
            line=dict(width=3)
        ))

        fig.add_trace(go.Scatter(
            x=df_forecast["Bird Age (days)"],
            y=df_forecast["Ideal Weight (kg)"],
            mode="lines+markers",
            name="Ideal Weight",
            line=dict(width=3, dash="dash")
        ))

        fig.update_layout(
            xaxis_title="Bird Age (Days)",
            yaxis_title="Weight (kg)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3)
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed — see logs for details.")
        st.error(str(e))
        traceback.print_exc()
