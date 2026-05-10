"""pages/forecast.py — Demand Forecast deep-dive"""
import streamlit as st
import pandas as pd
import numpy as np
from components.data_loader import load_forecast_data
from components.charts import forecast_line, residual_chart


def show():
    st.markdown("""
    <h1 style='font-family:Space Mono,monospace;font-size:1.8rem;
               color:#e2e8f0;letter-spacing:-1px;'>📈 Demand Forecast</h1>
    <p style='color:#64748b;font-size:0.9rem;margin-top:4px;'>
        Compare actual vs predicted demand. Filter by store, product, and date range.
    </p><br>
    """, unsafe_allow_html=True)

    df = load_forecast_data()

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            stores = ["All"] + sorted(df["store_id"].unique().tolist())
            sel_store = st.selectbox("Store", stores)
        with col2:
            products = ["All"] + sorted(df["product_id"].unique().tolist())
            sel_product = st.selectbox("Product", products)
        with col3:
            date_min = df["date"].min().date()
            date_max = df["date"].max().date()
            date_range = st.date_input("Date Range",
                                        value=(date_min, date_max),
                                        min_value=date_min, max_value=date_max)

    filtered = df.copy()
    if sel_store != "All":
        filtered = filtered[filtered["store_id"] == sel_store]
    if sel_product != "All":
        filtered = filtered[filtered["product_id"] == sel_product]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["date"] >= pd.Timestamp(date_range[0])) &
            (filtered["date"] <= pd.Timestamp(date_range[1]))
        ]
    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        st.warning("No data found for selected filters.")
        return

    # ── Sample Size Slider ────────────────────────────────────────────────────
    max_n = min(len(filtered), 2000)
    n_show = st.slider("Show last N samples", 100, max_n, min(500, max_n), step=50)

    # ── Main Forecast Chart ───────────────────────────────────────────────────
    st.plotly_chart(forecast_line(filtered, n=n_show), use_container_width=True)

    # ── Residual Chart ────────────────────────────────────────────────────────
    if "residual" in filtered.columns:
        st.plotly_chart(residual_chart(filtered.tail(n_show).reset_index(drop=True)),
                        use_container_width=True)

    # ── Per-Model Metrics ─────────────────────────────────────────────────────
    st.markdown("### 📊 Model Performance on Current Filter")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    for col_name, label in [("catboost_prediction","CatBoost"),
                              ("lightgbm_prediction","LightGBM"),
                              ("ensemble_prediction","Ensemble")]:
        if col_name in filtered.columns:
            yt = filtered["demand"]
            yp = filtered[col_name]
            rmse = float(np.sqrt(mean_squared_error(yt, yp)))
            mae  = float(mean_absolute_error(yt, yp))
            mask = yt != 0
            mape = float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100)
            r2   = float(r2_score(yt, yp))
            rows.append({"Model": label, "MAE": round(mae,2),
                         "RMSE": round(rmse,2), "MAPE (%)": round(mape,2),
                         "R²": round(r2,4), "Accuracy (%)": round(100-mape,2)})

    if rows:
        perf_df = pd.DataFrame(rows)
        st.dataframe(
            perf_df.style
            .highlight_min(subset=["RMSE","MAE","MAPE (%)"], color="#1f3b2e")
            .highlight_max(subset=["Accuracy (%)","R²"], color="#1f3b2e")
            .format({"RMSE":"{:.2f}","MAE":"{:.2f}","MAPE (%)":"{:.2f}%",
                     "R²":"{:.4f}","Accuracy (%)":"{:.2f}%"}),
            use_container_width=True, hide_index=True
        )

    # ── Data Preview ──────────────────────────────────────────────────────────
    with st.expander("🗃️ Raw Data Preview"):
        show_cols = ["date","store_id","product_id","demand",
                     "ensemble_prediction","residual"]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols].tail(200), use_container_width=True)

        csv = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download Filtered CSV", csv,
                           "filtered_forecast.csv", "text/csv")
