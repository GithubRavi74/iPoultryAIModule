# iPoultry AI Module — Weight, Mortality & FCR Prediction

## Overview

The **iPoultry AI Module** is designed to help poultry farm managers and agribusinesses make **data-driven decisions** by predicting:

- **Average Weight** of birds per batch
- **Daily Mortality**
- **Feed Conversion Ratio (FCR)**

using historical farm and batch data. This AI-powered module enables **optimized feeding, early risk detection, and operational efficiency** in poultry farming.

---

## Business Value

| Metric | Why It Matters | How AI Helps |
|--------|----------------|-------------|
| **Weight Prediction** | Determines growth performance & market readiness | Forecasts batch weight to adjust feed, plan sales, and manage inventory |
| **Mortality Prediction** | Minimizes losses and improves flock health | Identifies potential health or management issues in advance, enabling corrective actions |
| **FCR Prediction** | Measures feed efficiency | Helps optimize feed utilization, reducing costs and improving profitability |

**Impact**:

- Improve overall yield by up to **10–15%** through better feed planning
- Reduce unexpected bird mortality with **early alerts**
- Optimize feed costs and improve profitability via FCR monitoring
- Support farm managers with actionable insights **without manual calculations**

---

## Module Workflow

1. **Data Collection**:  
   Daily sensor data, feed/water logs, batch details, and environmental metrics (temperature, humidity, gas concentrations) are collected.

2. **ETL (Extract, Transform, Load)**:  
   Raw data is transformed into **ML-ready batch summaries**:
   - Aggregated feed, water consumption, and mortality per batch
   - Environmental averages over 1 day and 7-day rolling windows
   - Lag features for historical trends

3. **Model Training**:  
   Using historical batch summaries, the module trains **RandomForestRegressor pipelines** for:
   - Weight prediction
   - Mortality prediction
   - FCR prediction

   These models include preprocessing steps to scale numeric inputs, ensuring raw farm data can be used directly.

4. **Prediction / Deployment**:  
   The trained models are saved as **pipeline pickles** (`weight_pipeline.pkl`, `mortality_pipeline.pkl`, `fcr_pipeline.pkl`).  
   Predictions can be obtained via:
   - **Streamlit UI** (for quick interactive testing)
   - **Node.js backend API** (for integration with web or mobile dashboards)
   - **Direct Python scripts** (for automation or batch processing)

5. **Business Dashboard** (Optional):  
   Predictions can feed into dashboards for:
   - Daily batch monitoring
   - Feed efficiency reports
   - Mortality alerts
   - Exportable CSV for record-keeping and KPI tracking

---

## Input Features

The module uses **24 features per batch** including:

| Category | Features |
|----------|----------|
| Batch Info | `age_in_days`, `birds_alive`, `mortality` |
| Feed / Water | `feed_kg`, `water_consumption_l`, `feed_kg_7d`, `feed_kg_lag1/2/3` |
| Environmental | `avg_temp_c`, `avg_rh`, `avg_co_ppm`, `avg_nh_ppm`, and 7-day rolling averages |
| Bird Performance | `sample_weight`, `sample_weight_7d`, `fcr_today`, `fcr_today_7d` |
| Historical Mortality | `mortality_lag1`, `mortality_lag2`, `mortality_lag3` |

> The models are trained on processed historical data, so **users can input raw farm metrics** without manual feature engineering.

---

## Outputs

- **Predicted Weight** (kg) per batch
- **Predicted Daily Mortality** (birds)
- **Predicted Feed Conversion Ratio (FCR)**

The predictions are **numerical and actionable**, supporting operational and financial decisions.

---

## Technical Details

- **Model Type**: RandomForestRegressor with Scikit-learn Pipelines  
- **Preprocessing**: StandardScaler for numeric features  
- **Python Version Tested**: 3.11.x  
- **Scikit-Learn Version**: 1.3.2  
- **NumPy Version**: 1.26.x  
- **Pickle Format**: Pipeline pickles (`*_pipeline.pkl`) include preprocessing + model

**Why Pipelines:**  
- Accept raw inputs directly (no manual feature engineering needed)  
- Ensure feature ordering and scaling match training  
- Fully compatible with Node.js / Streamlit / Python scripts

---

## Quick Start (Streamlit Demo)

1. Place pipeline files in your working folder:
   - `weight_pipeline.pkl`  
   - `mortality_pipeline.pkl`  
   - `fcr_pipeline.pkl`  

2. Install Python dependencies:
```bash
pip install streamlit pandas numpy scikit-learn==1.3.2 cloudpickle
