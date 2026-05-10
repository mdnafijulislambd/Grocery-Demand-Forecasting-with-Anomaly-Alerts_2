"""pages/predictor.py — Live single-row prediction"""
import streamlit as st
import pandas as pd
import numpy as np
from components.data_loader import (load_catboost, load_lightgbm,
                                     load_label_encoders, load_feature_cols,
                                     load_forecast_data)

SURFACE = "#131929"; SURFACE2 = "#1a2236"; BORDER = "#243047"
ACCENT = "#00d4aa"; RED = "#ff6b6b"; YELLOW = "#ffd166"
TEXT = "#e2e8f0"; MUTED = "#64748b"


def _result_card(label, value, color=ACCENT):
    return f"""
    <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:14px;
                padding:1.3rem 1.5rem;border-top:3px solid {color};text-align:center;'>
        <div style='font-size:0.72rem;color:{MUTED};font-family:Space Mono,monospace;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;'>{label}</div>
        <div style='font-size:2.2rem;font-weight:700;color:{color};
                    font-family:Space Mono,monospace;'>{value}</div>
    </div>"""


def show():
    st.markdown("""
    <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;
               color:#e2e8f0;letter-spacing:-1px;'>🔮 Live Predictor</h1>
    <p style='color:#64748b;font-size:0.9rem;margin-top:4px;'>
        Input real-time features and get instant demand forecast from all models.
    </p><br>
    """, unsafe_allow_html=True)

    cat_model    = load_catboost()
    lgb_model    = load_lightgbm()
    encoders     = load_label_encoders()
    feature_cols = load_feature_cols()
    df_ref       = load_forecast_data()

    if cat_model is None and lgb_model is None:
        st.error("❌ No model files found in `models/` folder.")
        st.info("Upload `catboost_model.pkl` and `lightgbm_model.pkl` to continue.")
        return

    if not feature_cols:
        st.error("❌ `feature_cols.pkl` not found. Cannot determine input features.")
        return

    st.markdown("""
    <div style='background:#1a2236;border:1px solid #243047;border-radius:12px;
                padding:1rem 1.4rem;margin-bottom:1.5rem;'>
        <span style='color:#00d4aa;font-family:Space Mono,monospace;font-size:0.8rem;'>
            ℹ️  Fill in the inputs below. Lag/rolling features are auto-estimated from historical data if left at 0.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.markdown("#### 🏪 Store & Product")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            store_id   = st.selectbox("Store ID", sorted(df_ref["store_id"].unique()))
        with c2:
            product_id = st.selectbox("Product ID", sorted(df_ref["product_id"].unique()))
        with c3:
            category_vals = df_ref["category"].unique() if "category" in df_ref.columns else [0]
            category = st.selectbox("Category", sorted(category_vals))
        with c4:
            region_vals = df_ref["region"].unique() if "region" in df_ref.columns else [0]
            region = st.selectbox("Region", sorted(region_vals))

        st.markdown("#### 📅 Date")
        d1, d2, d3, d4 = st.columns(4)
        import datetime
        today = datetime.date.today()
        with d1: sel_date = st.date_input("Date", today)
        with d2: is_weekend = st.checkbox("Is Weekend", value=sel_date.weekday() >= 5)
        with d3: is_holiday = st.checkbox("Is Holiday", value=False)
        with d4: pass

        st.markdown("#### 💰 Pricing & Promotions")
        p1, p2, p3, p4 = st.columns(4)
        with p1: price             = st.number_input("Price",              0.0, 10000.0, 50.0, step=1.0)
        with p2: discount          = st.number_input("Discount (%)",       0.0, 100.0,   0.0,  step=1.0)
        with p3: competitor_pricing= st.number_input("Competitor Price",   0.0, 10000.0, 48.0, step=1.0)
        with p4: promotion         = st.selectbox("Promotion",             [0, 1])

        st.markdown("#### 🌤️ External Conditions")
        e1, e2, e3 = st.columns(3)
        with e1:
            weather_vals = df_ref["weather_condition"].unique() if "weather_condition" in df_ref.columns else [0]
            weather_condition = st.selectbox("Weather", sorted(weather_vals))
        with e2:
            season_vals = df_ref["seasonality"].unique() if "seasonality" in df_ref.columns else [0]
            seasonality = st.selectbox("Seasonality", sorted(season_vals))
        with e3:
            epidemic = st.selectbox("Epidemic", [0, 1])

        st.markdown("#### 📦 Inventory")
        i1, i2 = st.columns(2)
        with i1: inventory_level = st.number_input("Inventory Level", 0, 100000, 500, step=10)
        with i2: units_sold_ref  = st.number_input("Recent Units Sold (ref)", 0, 100000, 100, step=5)

        st.markdown("#### 📊 Historical Demand (Auto-filled if 0)")
        h1, h2, h3, h4, h5 = st.columns(5)
        with h1: lag_1  = st.number_input("Lag 1 (yesterday)", 0.0, 10000.0, 0.0)
        with h2: lag_2  = st.number_input("Lag 2",             0.0, 10000.0, 0.0)
        with h3: lag_7  = st.number_input("Lag 7",             0.0, 10000.0, 0.0)
        with h4: lag_14 = st.number_input("Lag 14",            0.0, 10000.0, 0.0)
        with h5: lag_30 = st.number_input("Lag 30",            0.0, 10000.0, 0.0)

        submitted = st.form_submit_button("🔮 Predict Demand", use_container_width=True)

    if not submitted:
        return

    # ── Auto-fill lags from historical data ───────────────────────────────────
    hist = df_ref[(df_ref["store_id"] == store_id) &
                  (df_ref["product_id"] == product_id)].tail(30)

    def _auto(user_val, hist_vals, default=100.0):
        if user_val > 0:
            return float(user_val)
        return float(hist_vals.mean()) if len(hist_vals) > 0 else default

    l1  = _auto(lag_1,  hist["demand"].tail(1))
    l2  = _auto(lag_2,  hist["demand"].tail(2))
    l7  = _auto(lag_7,  hist["demand"].tail(7))
    l14 = _auto(lag_14, hist["demand"].tail(14))
    l30 = _auto(lag_30, hist["demand"].tail(30))

    rm7  = hist["demand"].tail(7).mean()  if len(hist) >= 7  else l1
    rm14 = hist["demand"].tail(14).mean() if len(hist) >= 14 else l1
    rm30 = hist["demand"].tail(30).mean() if len(hist) >= 30 else l1
    rs7  = hist["demand"].tail(7).std()   if len(hist) >= 7  else 0
    rs30 = hist["demand"].tail(30).std()  if len(hist) >= 30 else 0
    exp_mean = hist["demand"].mean()      if len(hist) > 0   else l1

    dt = pd.Timestamp(sel_date)

    row = {
        "store_id": store_id, "product_id": product_id,
        "category": category, "region": region,
        "year": dt.year, "month": dt.month, "day": dt.day,
        "day_of_week": dt.dayofweek,
        "week_of_year": dt.isocalendar().week,
        "quarter": dt.quarter,
        "is_weekend": int(is_weekend), "is_holiday": int(is_holiday),
        "day_name": dt.day_name(),
        "month_sin": np.sin(2*np.pi*dt.month/12),
        "month_cos": np.cos(2*np.pi*dt.month/12),
        "dow_sin":   np.sin(2*np.pi*dt.dayofweek/7),
        "dow_cos":   np.cos(2*np.pi*dt.dayofweek/7),
        "price": price, "discount": discount,
        "competitor_pricing": competitor_pricing,
        "promotion": promotion,
        "weather_condition": weather_condition,
        "seasonality": seasonality, "epidemic": epidemic,
        "inventory_level": inventory_level,
        "demand_lag_1": l1, "demand_lag_2": l2, "demand_lag_7": l7,
        "demand_lag_14": l14, "demand_lag_30": l30,
        "demand_change_1": l1 - l2,
        "demand_change_7": l7 - l14,
        "rolling_mean_7": rm7, "rolling_mean_14": rm14, "rolling_mean_30": rm30,
        "rolling_std_7": rs7, "rolling_std_30": rs30,
        "expanding_mean": exp_mean,
        "inventory_sales_ratio": inventory_level / (units_sold_ref + 1),
        "price_diff": price - competitor_pricing,
        "discounted_price": price * (1 - discount/100),
    }

    # Encode categoricals
    for col in ["store_id","product_id","category","region",
                "weather_condition","seasonality","day_name"]:
        if col in encoders and col in row:
            try:
                row[col] = encoders[col].transform([str(row[col])])[0]
            except ValueError:
                row[col] = 0

    input_df = pd.DataFrame([row])

    # Align to feature_cols
    for fc in feature_cols:
        if fc not in input_df.columns:
            input_df[fc] = 0
    input_df = input_df[feature_cols]

    # ── Predictions ───────────────────────────────────────────────────────────
    cat_pred = float(cat_model.predict(input_df)[0]) if cat_model else None
    lgb_pred = float(lgb_model.predict(input_df)[0]) if lgb_model else None

    if cat_pred and lgb_pred:
        total = (1/cat_pred + 1/lgb_pred) if cat_pred and lgb_pred else 2
        ens_pred = (cat_pred + lgb_pred) / 2
    elif cat_pred:
        ens_pred = cat_pred
    else:
        ens_pred = lgb_pred

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🎯 Prediction Results")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(_result_card("CatBoost", f"{cat_pred:.1f}" if cat_pred else "N/A", ACCENT),
                    unsafe_allow_html=True)
    with rc2:
        st.markdown(_result_card("LightGBM", f"{lgb_pred:.1f}" if lgb_pred else "N/A", "#74b9ff"),
                    unsafe_allow_html=True)
    with rc3:
        st.markdown(_result_card("Ensemble", f"{ens_pred:.1f}", YELLOW),
                    unsafe_allow_html=True)

    # ── Confidence Band ───────────────────────────────────────────────────────
    if cat_pred and lgb_pred:
        diff = abs(cat_pred - lgb_pred)
        low  = min(cat_pred, lgb_pred) * 0.9
        high = max(cat_pred, lgb_pred) * 1.1
        agreement = "✅ High" if diff < ens_pred * 0.1 else ("⚠️ Medium" if diff < ens_pred * 0.25 else "❌ Low")

        st.markdown(f"""
        <br>
        <div style='background:{SURFACE2};border:1px solid {BORDER};border-radius:14px;
                    padding:1.3rem 1.5rem;'>
            <div style='font-family:Space Mono,monospace;font-size:0.78rem;color:{MUTED};
                        margin-bottom:12px;'>PREDICTION CONFIDENCE BAND</div>
            <div style='display:flex;gap:2rem;font-size:0.92rem;'>
                <span>📉 <b>Lower</b>: {low:.1f}</span>
                <span>📈 <b>Upper</b>: {high:.1f}</span>
                <span>🤝 <b>Model Agreement</b>: {agreement}</span>
                <span>📏 <b>Spread</b>: ±{diff/2:.1f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input Summary ─────────────────────────────────────────────────────────
    with st.expander("🔍 View Input Summary"):
        st.dataframe(pd.DataFrame([row]), use_container_width=True)
