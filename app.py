import streamlit as st

st.set_page_config(
    page_title="GroceryAI Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:        #0b0f19;
    --surface:   #131929;
    --surface2:  #1a2236;
    --border:    #243047;
    --accent:    #00d4aa;
    --accent2:   #ff6b6b;
    --accent3:   #ffd166;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00b894) !important;
    color: #0b0f19 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,212,170,0.3) !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: var(--font-mono) !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px !important; }

/* Selectbox / sliders */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Headers */
h1, h2, h3 { font-family: var(--font-mono) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Alert boxes */
[data-testid="stAlert"] {
    background: var(--surface2) !important;
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 2rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.4rem;
                    font-weight: 700; color: #00d4aa; letter-spacing: -1px;'>
            🛒 GroceryAI
        </div>
        <div style='font-size: 0.75rem; color: #64748b; margin-top: 4px;
                    font-family: Space Mono, monospace;'>
            Demand · Forecast · Anomaly
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["🏠 Overview", "📈 Demand Forecast", "🚨 Anomaly Detection",
         "📊 Feature Insights", "🔮 Live Predictor"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.72rem; color: #64748b; font-family: Space Mono, monospace;
                line-height: 1.8;'>
        Models<br>
        <span style='color:#00d4aa'>■</span> CatBoost<br>
        <span style='color:#ff6b6b'>■</span> LightGBM<br>
        <span style='color:#ffd166'>■</span> Ensemble<br><br>
        Anomaly Detection<br>
        <span style='color:#00d4aa'>■</span> LOF + Z-Score
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.68rem; color: #3d4f6e; font-family: Space Mono, monospace;'>
        v1.0.0 · FIXED MODEL<br>
        No Data Leakage ✓
    </div>
    """, unsafe_allow_html=True)

# ── Page Routing ─────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    from pages.overview import show
    show()
elif page == "📈 Demand Forecast":
    from pages.forecast import show
    show()
elif page == "🚨 Anomaly Detection":
    from pages.anomaly import show
    show()
elif page == "📊 Feature Insights":
    from pages.features import show
    show()
elif page == "🔮 Live Predictor":
    from pages.predictor import show
    show()
