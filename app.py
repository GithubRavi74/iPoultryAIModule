# -----------------------------
# Prediction and Display
# -----------------------------
if submitted:

    # number of days to forecast
    forecast_days = 33

    # Create future age sequence
    future_ages = np.arange(age_in_days, age_in_days + forecast_days)

    # Empty lists to store predictions
    weight_preds = []
    mort_preds = []
    fcr_preds = []

    # Loop through each future day
    for future_age in future_ages:

        row = pd.DataFrame([{
            "age_in_days": future_age,
            "birds_alive": birds_alive,
            "mortality": mortality_today,
            "feed_kg": feed_today,
            "water_consumption_l": water_today,
            "avg_temp_c": 28,     # placeholder
            "avg_rh": 60,
            "avg_co_ppm": 400,
            "avg_nh_ppm": 18,
            "sample_weight": sample_weight_today,
            "fcr_today": feed_today / sample_weight_today if sample_weight_today > 0 else 0,

            # same historical placeholders
            "avg_temp_c_7d": 28,
            "avg_rh_7d": 60,
            "avg_co_ppm_7d": 400,
            "avg_nh_ppm_7d": 18,
            "feed_kg_7d": feed_today * 7,
            "sample_weight_7d": sample_weight_today,
            "fcr_today_7d": feed_today / sample_weight_today if sample_weight_today > 0 else 0,
            "mortality_lag1": mortality_today,
            "mortality_lag2": mortality_today,
            "mortality_lag3": mortality_today,
            "feed_kg_lag1": feed_today,
            "feed_kg_lag2": feed_today,
            "feed_kg_lag3": feed_today
        }])

        row_w = row[weight_model.feature_names_in_]
        row_m = row[mortality_model.feature_names_in_]
        row_f = row[fcr_model.feature_names_in_]

        weight_preds.append(weight_model.predict(row_w)[0])
        mort_preds.append(mortality_model.predict(row_m)[0])
        fcr_preds.append(fcr_model.predict(row_f)[0])

    # Build dataframe for display
    df_forecast = pd.DataFrame({
        "Day (Bird Age)": future_ages,
        "Predicted Weight (kg)": np.round(weight_preds, 3),
        "Predicted Mortality": np.round(mort_preds).astype(int),
        "Predicted FCR": np.round(fcr_preds, 3)
    })

    # Display
    st.subheader("📈 33-Day Forecast (Based on Today's Age)")
    st.dataframe(df_forecast, use_container_width=True)

    # Plot Weight Prediction
    st.markdown("### Weight Forecast Curve")
    st.line_chart(df_forecast.set_index("Day (Bird Age)")["Predicted Weight (kg)"])
