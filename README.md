# 🛒 GroceryAI — Demand Forecasting & Anomaly Detection Dashboard

Streamlit dashboard for grocery store demand forecasting using CatBoost + LightGBM ensemble with LOF + Z-Score anomaly detection.

## 📁 Project Structure

```
grocery_dashboard/
├── app.py                        # Main entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml               # Dark theme config
├── pages/
│   ├── overview.py               # KPI overview
│   ├── forecast.py               # Demand forecast charts
│   ├── anomaly.py                # Anomaly detection
│   ├── features.py               # Feature importance
│   └── predictor.py              # Live single-row predictor
├── components/
│   ├── data_loader.py            # Cached data/model loaders
│   └── charts.py                 # Reusable Plotly chart builders
├── models/                       # ← Put your .pkl files here
│   ├── catboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── label_encoders.pkl
│   └── feature_cols.pkl
└── data/                         # ← Put your .csv files here
    ├── forecast_dashboard_data.csv
    └── forecast_anomaly_results.csv
```

## 🚀 Setup & Deploy

### Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → set **Main file**: `app.py`
4. Click **Deploy**

> ⚠️ **Important**: Upload your model files and CSV files before deploying.
> Do NOT commit large `.pkl` files to GitHub — use Streamlit Secrets or upload via the app.

## 📊 Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, daily trend, store/product analysis |
| 📈 Demand Forecast | Filter by store/product/date, compare models |
| 🚨 Anomaly Detection | LOF + Z-Score anomaly visualization & export |
| 📊 Feature Insights | Feature importance, leakage check, correlation |
| 🔮 Live Predictor | Real-time prediction with auto-filled historical lags |

## ✅ Model Fix Summary

The original notebook had **data leakage** in `demand_change` features.
This dashboard uses the **FIXED** model where:
- `demand_change_1 = lag_1 − lag_2` (not `demand − lag_1`)
- `demand_change_7 = lag_7 − lag_14` (not `demand − lag_7`)
- `units_sold` excluded from feature list
